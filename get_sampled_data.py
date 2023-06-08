# %%
import pickle
import json
from tqdm import tqdm
import datasets
import tiktoken
import random
random.seed(42)

TOKENIZER_MODEL_ALIAS_MAP = {
    "gpt-4": "gpt-3.5-turbo",
    "gpt-35-turbo": "gpt-3.5-turbo",
}
query_data: dict = json.load(open("resources/codequeries_meta.json", "r"))
reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data}
all_queries = reccos_dict.keys()


def count_file_tokens(file_path: str, model_name: str = "gpt-35-turbo"):
    with open('/home/t-susahu/CodeQueries/data' + f'/{file_path}', 'r') as f:
        input_str = f.read()
    if model_name in TOKENIZER_MODEL_ALIAS_MAP:
        model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_str))
    return {'file_tokens': num_tokens}


# To get file which can be fit into prompt, data from which `sampled test data` is created
def generate_querywise_test_samples():
    dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
    partitioned_data_all = {query['name']: dataset.map(lambda x: count_file_tokens(x['code_file_path'])).filter(lambda x: x["query_name"] == query['name'] and x["file_tokens"] < 3000) for query in query_data}

    with open('resources/partitioned_data_all.pkl', 'wb') as f:
        pickle.dump(partitioned_data_all, f)


# To get sampled 20 files for twostep train
def generate_querywise_train_all_samples():
    dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TRAIN)
    partitioned_data_train_all = {query['name']: dataset.filter(lambda x: x["query_name"] == query['name']) for query in query_data}

    with open('resources/partitioned_data_train_all.pkl', 'wb') as f:
        pickle.dump(partitioned_data_train_all, f)


# To get train files to be used as examples with LLM prompting. In case you want run LLM experiment with supporting fact prompting, you need to run this.
def generate_querywise_train_samples():
    dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TRAIN)
    partitioned_data_train_1000 = {query['name']: dataset.map(lambda x: count_file_tokens(x['code_file_path'])).filter(lambda x: x["query_name"] == query['name'] and x["file_tokens"] < 1000) for query in query_data}

    with open('resources/partitioned_data_train_1000.pkl', 'wb') as f:
        pickle.dump(partitioned_data_train_1000, f)


# To get how many files after sampling
def get_querywise_test_stat(partitioned_data_path):
    dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
    with open(partitioned_data_path, 'rb') as f:
        partitioned_data_all = pickle.load(f)

    all_files = dataset['code_file_path']
    total = 0
    for query in all_queries:
        total += partitioned_data_all[query].shape[0]
        for ff in partitioned_data_all[query]['code_file_path']:
            assert ff in all_files

    print(dataset.shape[0], total)


def sample_data(query_data, split, s=10):
    if split == 'train':
        pos_files = set(query_data.filter(lambda x: x["example_type"] == 1)['code_file_path'])
        neg_files = set(query_data.filter(lambda x: x["example_type"] == 0)['code_file_path'])
    else:
        pos_files = set(query_data.filter(lambda x: x["example_type"] == 1 and x['file_tokens'] <= 2000)['code_file_path'])
        neg_files = set(query_data.filter(lambda x: x["example_type"] == 0 and x['file_tokens'] <= 2000)['code_file_path'])
    assert len(pos_files.intersection(neg_files)) == 0

    all_files = random.sample(pos_files, min(s, len(pos_files))) + random.sample(neg_files, min(s, len(neg_files)))

    # get answer and sf spans
    metadata_with_spans = {}
    for ff in all_files:
        ans_spans = []
        sf_spans = []
        file_data = query_data.filter(lambda x: x['code_file_path'] == ff and x["example_type"] == 1)
        for row in file_data:
            ans_spans += row['answer_spans']
            sf_spans += row['supporting_fact_spans']
        metadata_with_spans[ff] = {'ans_spans': ans_spans, 'sf_spans': sf_spans}
        if ff in neg_files:
            assert not ans_spans and not sf_spans

    return metadata_with_spans


# From partitioned data, get metadata for `test sampled data` and sampled max 20 train files for two-step.
# Output for this function is provided, which can be directly used with `get_instance_foe_files`.
def get_sampled_file_metadata(partitioned_data_path, output_path, split='train'):
    with open(partitioned_data_path, 'rb') as f:
        partitioned_data = pickle.load(f)

    sampled_partitioned_data = {}
    for query in tqdm(all_queries):
        sampled_partitioned_data[query] = sample_data(partitioned_data[query], split)

    with open(output_path, 'wb') as f:
        pickle.dump(sampled_partitioned_data, f)


# To get the twostep data for `sampled test data` or sampled max 20 train files
def get_instances_for_files(sampled_data_path, dataset_setting, split):
    if split == 'TEST':
        dataset = datasets.load_dataset("thepurpleowl/codequeries", dataset_setting, split=datasets.Split.TEST)
    else:
        dataset = datasets.load_dataset("thepurpleowl/codequeries", dataset_setting, split=datasets.Split.TRAIN)
    with open(sampled_data_path, 'rb') as f:
        sampled_data = pickle.load(f)

    filtered_data = dataset.filter(lambda x: x['code_file_path'] in list(sampled_data[x["query_name"]].keys()))
    with open(f'resources/{dataset_setting}_{split}.pkl', 'wb') as f:
        pickle.dump(filtered_data, f)


if __name__ == '__main__':
    # get_sampled_file_metadata('resources/partitioned_data_all.pkl', 'resources/sampled_test_data.pkl', 'test')
    get_instances_for_files('resources/sampled_test_data.pkl', 'twostep', 'TEST')
