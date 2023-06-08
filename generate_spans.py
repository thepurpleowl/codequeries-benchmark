from pathlib import Path
import pickle
import json
import os
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from utils_openai import OpenAIModelHandler, PromptConstructor, LogWriter


logging.basicConfig(filename=f'request_logs_{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger('api_logger')


def get_sanitized_content(ctxt_blocks):
    context_blocks = ""
    for ctxt_block in ctxt_blocks:
        newline_removed_content = ("\n".join(line 
                                             for line in ctxt_block['content'].split('\n')
                                             if line))
        context_blocks += newline_removed_content
        context_blocks += '\n'
    return context_blocks


def get_examples_values(query, pos_ex, neg_ex):
    y = {}
    assert query == pos_ex['query_name'] == neg_ex['query_name']
    y['positive_context'] = get_sanitized_content(pos_ex['context_blocks'])
    y['positive_spans'] = [ans['span'] for ans in pos_ex['answer_spans']]
    y['negative_context'] = get_sanitized_content(neg_ex['context_blocks'])

    return y


def get_sf_examples_values(ex, prompt_constructor, with_sf):
    y = {}
    y['positive_context'] = get_sanitized_content(ex['context_blocks'])
    y['positive_spans'] = [ans['span'] for ans in ex['answer_spans']]
    if with_sf:
        y['supporting_fact_spans'] = [ans['span'] for ans in ex['supporting_fact_spans']]

    return prompt_constructor.construct(**y)


def get_file_level_prompt_input(query, desc_dict, file_path, partitioned_data_all):
    df = partitioned_data_all[query].filter(lambda example: example["code_file_path"] == file_path and example['query_name'] == query)
    y = {}
    y['query_name'] = query
    y['description'] = desc_dict[query]
    with open(f"/home/t-susahu/CodeQueries/data/{file_path}", 'r') as f:
        y['input_code'] = f.read()
    y['answer_spans'] = ''
    y['supporting_fact_spans'] = ''
    for row in df:
        y['answer_spans'] += ':::-:::'.join([ans['span'] for ans in row['answer_spans']])
        y['supporting_fact_spans'] += ':::-:::'.join([sf['span'] for sf in row['supporting_fact_spans']])
    if y['answer_spans']:
        for row in df:
            assert row['example_type'] == 1
    else:
        for row in df:
            assert row['example_type'] == 0
        y['answer_spans'] = 'N/A'

    if not y['supporting_fact_spans']:
        y['supporting_fact_spans'] = 'N/A'
    return y


def get_file_level_prompt_input_from_metadata(query, desc_dict, file_path, sampled_querywise_files):
    y = {}
    y['query_name'] = query
    y['description'] = desc_dict[query]
    with open(f"/home/t-susahu/CodeQueries/data/{file_path}", 'r') as f:
        y['input_code'] = f.read()
    y['answer_spans'] = ':::-:::'.join([ans['span'] for ans in sampled_querywise_files[query][file_path]['ans_spans']])
    y['supporting_fact_spans'] = ':::-:::'.join([sf['span'] for sf in sampled_querywise_files[query][file_path]['sf_spans']])
    if not y['answer_spans']:
        y['answer_spans'] = 'N/A'

    if not y['supporting_fact_spans']:
        y['supporting_fact_spans'] = 'N/A'
    return y


def get_filename(fn):
    return '_'.join(fn.split('/'))


def extract_codeql_recommendation(meta_data):
    query_data: dict = json.load(open(meta_data, "r", encoding="utf-8"))
    reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data}
    desc_dict: dict = {q["name"]: q["desc"] for q in query_data}

    return query_data, reccos_dict, desc_dict


def get_random_examples_for_prompt(pos_exes, neg_exes, query):
    pos_ex = pos_exes.shuffle().select([0])
    neg_ex = neg_exes.shuffle().select([0])
    assert not neg_ex[0]['answer_spans']
    assert not neg_ex[0]['supporting_fact_spans']

    return pos_ex[0], neg_ex[0]


def get_retrieved_examples_for_prompt(pos_exes, neg_exes, bm25_db_pos, bm25_db_neg, input_code):
    tokenized_input_code = input_code.split(" ")

    pos_ex_scores = bm25_db_pos.get_scores(tokenized_input_code)
    neg_ex_scores = bm25_db_neg.get_scores(tokenized_input_code)
    selected_pos_index = int(pos_ex_scores.argmax())
    selected_neg_index = int(neg_ex_scores.argmax())
    assert not neg_exes[selected_neg_index]['answer_spans']
    assert not neg_exes[selected_neg_index]['supporting_fact_spans']

    return pos_exes[selected_pos_index], neg_exes[selected_neg_index]


def get_retrieved_sf_examples_for_prompt(pos_exes_with_sf, pos_exes_wo_sf, bm25_db_pos_with_sf, bm25_db_pos_wo_sf, 
                                         input_code, **sf_constructors):
    tokenized_input_code = input_code.split(" ")
    example_sf_description = ""
    if not bm25_db_pos_with_sf:
        example_sf_description = "The query examples do not have any supporting fact span."
        ex_scores = bm25_db_pos_wo_sf.get_scores(tokenized_input_code)
        selected_indices = [int(x) for x in (-ex_scores).argsort()[:2]]
        ex_a = pos_exes_wo_sf[selected_indices[0]]
        ex_b = pos_exes_wo_sf[selected_indices[1]]
        ex_a_text = get_sf_examples_values(ex_a, sf_constructors['prompt_wo_sf'], False)
        ex_b_text = get_sf_examples_values(ex_b, sf_constructors['prompt_wo_sf'], False)
    elif not bm25_db_pos_wo_sf:
        ex_scores = bm25_db_pos_with_sf.get_scores(tokenized_input_code)
        selected_indices = [int(x) for x in (-ex_scores).argsort()[:2]]
        ex_a = pos_exes_with_sf[selected_indices[0]]
        ex_b = pos_exes_with_sf[selected_indices[1]]
        ex_a_text = get_sf_examples_values(ex_a, sf_constructors['prompt_with_sf'], True)
        ex_b_text = get_sf_examples_values(ex_b, sf_constructors['prompt_with_sf'], True)
    else:
        ex_with_sf_scores = bm25_db_pos_with_sf.get_scores(tokenized_input_code)
        ex_wo_sf_scores = bm25_db_pos_wo_sf.get_scores(tokenized_input_code)
        selected_with_sf_ex_index = int(ex_with_sf_scores.argmax())
        selected_wo_sf_ex_index = int(ex_wo_sf_scores.argmax())
        ex_a = pos_exes_with_sf[selected_with_sf_ex_index]
        ex_b = pos_exes_wo_sf[selected_wo_sf_ex_index]
        ex_a_text = get_sf_examples_values(ex_a, sf_constructors['prompt_with_sf'], True)
        ex_b_text = get_sf_examples_values(ex_b, sf_constructors['prompt_wo_sf'], False)

    example_input = {'example_sf_description': example_sf_description,
                     'example_a': ex_a_text,
                     'example_b': ex_b_text}
    return example_input


def get_bm25_db(exes): 
    tokenized_corpus = [get_sanitized_content(cbs).split(" ") for cbs in exes['context_blocks']]
    bm25_db = BM25Okapi(tokenized_corpus)

    return bm25_db


def run(few_shot, random_selection, with_sf,
        prompt_constructor, model_handler, experiment_config,
        **sf_constructors):
    with open('resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)
    with open('resources/sampled_test_data.pkl', 'rb') as f: 
        sampled_querywise_files = pickle.load(f)

    _, _, desc_dict = extract_codeql_recommendation("resources/codequeries_meta.json")

    if few_shot:
        with open('resources/partitioned_data_train_1000.pkl', 'rb') as f: 
            partitioned_data_train_1000 = pickle.load(f)

    Logger = LogWriter()
    all_queries = list(sampled_querywise_files.keys())
    for query in tqdm(all_queries):
        logger.info(f'Current query: {query}')
        query_folderName = query_folderName_map[query]
        if not Path(experiment_config["Target_folder"] + "/logs").exists():
            os.makedirs(experiment_config["Target_folder"] + "/logs")
        if not Path(experiment_config["Target_folder"] + f"/{query_folderName}").exists():
            os.makedirs(experiment_config["Target_folder"] + f"/{query_folderName}")

        sampled_files = list(sampled_querywise_files[query].keys())

        if few_shot and not random_selection:
            # (4000 - 200 - 300)/2 ~ 750
            if not with_sf:
                pos_exes = partitioned_data_train_1000[query].filter(lambda x: x["example_type"] == 1 and x["file_tokens"] < 700)
                neg_exes = partitioned_data_train_1000[query].filter(lambda x: x["example_type"] == 0 and x["file_tokens"] < 700)
                bm25_db_pos = get_bm25_db(pos_exes)
                bm25_db_neg = get_bm25_db(neg_exes)
            else:
                pos_exes = partitioned_data_train_1000[query].filter(lambda x: x["example_type"] == 1 and x["file_tokens"] < 700)
                pos_exes_with_sf = pos_exes.filter(lambda x: len(x["supporting_fact_spans"]) != 0)
                pos_exes_wo_sf = pos_exes.filter(lambda x: len(x["supporting_fact_spans"]) == 0)

                with_sf_count = pos_exes_with_sf.shape[0]
                wo_sf_count = pos_exes_wo_sf.shape[0]
                
                bm25_db_pos_with_sf = None
                bm25_db_pos_wo_sf = None
                if with_sf_count != 0:
                    bm25_db_pos_with_sf = get_bm25_db(pos_exes_with_sf)
                if wo_sf_count != 0:
                    bm25_db_pos_wo_sf = get_bm25_db(pos_exes_wo_sf)
                assert (bm25_db_pos_with_sf is not None) or (bm25_db_pos_wo_sf is not None)
        elif few_shot and random_selection:
            pos_exes = partitioned_data_train_1000[query].filter(lambda x: x["example_type"] == 1)
            neg_exes = partitioned_data_train_1000[query].filter(lambda x: x["example_type"] == 0)

        processed_rows = []
        i = 0
        for file_path in tqdm(sampled_files):
            prompt_input = get_file_level_prompt_input_from_metadata(query, desc_dict, file_path, sampled_querywise_files)
            if with_sf and prompt_input['answer_spans'] == 'N/A':
                continue

            if few_shot:
                if not random_selection:
                    if not with_sf:
                        pos_ex, neg_ex = get_retrieved_examples_for_prompt(pos_exes, neg_exes, bm25_db_pos, bm25_db_neg, prompt_input['input_code'])
                        example_values = get_examples_values(query, pos_ex, neg_ex)
                    else:
                        example_values = get_retrieved_sf_examples_for_prompt(pos_exes_with_sf, pos_exes_wo_sf,
                                                                              bm25_db_pos_with_sf, bm25_db_pos_wo_sf,
                                                                              prompt_input['input_code'],
                                                                              **sf_constructors)
                elif random_selection:
                    pos_ex, neg_ex = get_random_examples_for_prompt(pos_exes, neg_exes, query)
                    example_values = get_examples_values(query, pos_ex, neg_ex)

                for k in example_values:
                    prompt_input[k] = example_values[k]

            prompt_str = prompt_constructor.construct(**prompt_input)
            if not prompt_str:
                logger.info(file_path, prompt_str)
                continue

            with open(Path(experiment_config["Target_folder"] + f"/{query_folderName}/{get_filename(file_path)}.log"), 'w') as f:
                f.write(prompt_str)

            original_responses = model_handler.get_response(prompt_str)
            if with_sf:
                p_row = [i,
                         query,
                         file_path,
                         '',  # prompt_str,
                         prompt_input['answer_spans'],
                         prompt_input['supporting_fact_spans']]
            else:
                p_row = [i,
                         query,
                         file_path,
                         '',  # prompt_str,
                         prompt_input['answer_spans']]
            p_row.extend([original_responses[i].text.strip() if original_responses[i].text else '' for i in range(experiment_config['n'])])
            processed_rows.append(p_row)
            i += 1

        Logger.create_logs(experiment_config["Target_folder"] + f"/logs/{query_folderName}_logs.csv",
                           processed_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_path", "-k", type=str, default="resources/key_fast.txt")
    parser.add_argument("--api_type", type=str, choices=["azure", "open_ai"], default="azure")
    parser.add_argument("--api_base", type=str, default="https://gcrgpt4aoai6c.openai.azure.com/")
    parser.add_argument("--api_version", type=str, default="2023-03-15-preview")

    parser.add_argument("--Target_folder", "-t", type=str, required=True)  # "test_dir_file_fewshot"
    parser.add_argument("--prompt_template", "-p", type=str, required=True)  # "prompt_templates/span_highlight_fewshot.j2"

    parser.add_argument("--model", type=str, default='gpt-35-tunro')
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--stop", type=str, default="```")  # "```END" for 'with_sf=True' experiment
    parser.add_argument("--n", type=int, default=10)

    parser.add_argument("--few_shot", type=bool, default=False)
    parser.add_argument("--random_selection", type=bool, default=False)
    parser.add_argument("--with_sf", type=bool, default=False)

    args = parser.parse_args()
    # set config
    model_config = {
        "engine": args.model,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n": args.n,
        "stop": [args.stop]
    }

    experiment_config = {
        "Target_folder": args.Target_folder,
        "template_path": args.prompt_template,
        "encoding": 'UTF-8',
        "timeout": 8,
        "prompt_batch_size": 1,
        "timeout_mf": 2,
        "max_attempts": 10,
        "include_related_lines": True,
        "max_prompt_size": 4000,
        "extra_buffer": 200,
        "n": model_config["n"],
        "system_message": """Assistant is an AI chatbot that helps developers perform code quality related tasks.""",
    }

    model_handler = OpenAIModelHandler(
        model_config,
        Path(args.api_key_path).read_text().strip(),  # openai_api_key
        timeout=experiment_config["timeout"],
        prompt_batch_size=experiment_config["prompt_batch_size"],
        max_attempts=experiment_config["max_attempts"],
        timeout_mf=experiment_config["timeout_mf"],
        openai_api_type=args.api_type,
        openai_api_base=args.api_base,
        openai_api_version=args.api_version,
        system_message=experiment_config["system_message"]
    )

    prompt_constructor = PromptConstructor(
        template_path=experiment_config['template_path'],
        model=model_config["model"],
    )
    ex_a_prompt_constructor = PromptConstructor(
        template_path="prompt_templates/ex_with_sf.j2",
        model=model_config["model"],
    )
    ex_b_prompt_constructor = PromptConstructor(
        template_path="prompt_templates/ex_wo_sf.j2",
        model=model_config["model"],
    )
    sf_constructors = {'prompt_with_sf': ex_a_prompt_constructor,
                       'prompt_wo_sf': ex_b_prompt_constructor}

    # get response
    run(few_shot=args.few_shot, random_selection=args.random_selection, with_sf=args.with_sf,
        prompt_constructor=prompt_constructor,
        model_handler=model_handler,
        experiment_config=experiment_config,
        **sf_constructors)
