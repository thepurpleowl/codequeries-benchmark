import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import os
import csv
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
smoothing_fn = SmoothingFunction()


# from difflib import SequenceMatcher
__FILE_DIR__ = Path(__file__).parent
nck_dict = {}


def power_set(k):
    n = 10
    for i in range(2**n):
        combo = []
        for j in range(n):
            if (i >> j) % 2 == 1:
                combo.append(j)
        if len(combo) == k:
            yield combo


def build_nck():
    nck_dict = {}  # for nCk
    for k in [1, 2, 5, 10]:
        nck_dict[k] = list(power_set(k))
    return nck_dict


def cal_pass_at_k(n, c, k):
    if n - c < k:
        return 1
    return 1 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def singlespan_eq(actual_span, span):
    if actual_span.strip() == span.strip():
        return 1
    # else:
    #     if actual_span != 'N/A':
    #         match = SequenceMatcher(None, actual_span, span).find_longest_match()
    #         if (abs(match.size - len(actual_span)) < 3 and abs(len(actual_span) -len(span)) < 3):
    #             return 1

    return 0


def multispan_eq(actual_spans, spans):
    actual_spans = actual_spans.strip()
    spans = spans.strip()

    checked_generated_spans = {}
    for si, span in enumerate(spans.split('\n')):
        checked_generated_spans[si] = {'text': span, 'checked': False}

    for actual_span in actual_spans.split(':::-:::'):
        this_span_eq = False
        for si in checked_generated_spans:
            if(not checked_generated_spans[si]['checked']
                    and singlespan_eq(actual_span, checked_generated_spans[si]['text']) == 1):
                checked_generated_spans[si]['checked'] = True
                this_span_eq = True
                break
        # for some actual span if we dont find EM, then return 0
        if not this_span_eq:
            return 0

    # if for all actual spans we found EM, but extra spans present in generated span
    for si in checked_generated_spans:
        if not checked_generated_spans[si]['checked']:
            return 0

    return 1


def score_at_k(actual_spans, generated_spans, k):
    # print('Actual :', actual_spans)
    # print(generated_spans)
    # print('-'*50, len(generated_spans))

    sentence_bleu = 0
    for selected_indices in nck_dict[k]:
        selected_gen_spans = []
        for index in selected_indices:
            selected_gen_spans.append(generated_spans[index])

        bleu_scores = []
        for span in selected_gen_spans:
            candidate = span.split()
            reference = []
            for x in actual_spans.split(':::-:::'):
                reference.extend(x.split())
            bleu_scores.append(corpus_bleu([[reference]], [candidate],
                                           weights=(0.25, 0.25, 0.25, 0.25),
                                           smoothing_function=smoothing_fn.method2))
        # get max bleu from this combination
        sentence_bleu += max(bleu_scores)

    # normalize
    return sentence_bleu / len(nck_dict[k])


def pass_at_k(log_path, all_queries, query_folderName_map, k, example_type, n=10):
    pass_at_k = 0
    total = 0
    _cols_ = ['i', 'query', 'file_path', 'prompt', 'ans_spans']
    _cols_.extend([f'ans_{i}' for i in range(n)])
    querywise_em = {}
    corpus_sentence_bleu = 0
    for query in all_queries:
        prev_pass_at_k = pass_at_k
        try:
            df = pd.read_csv(__FILE_DIR__ / f'{log_path}/{query_folderName_map[query]}_logs.csv',
                             names=_cols_, keep_default_na=False)
            assert df.shape[0] == 20 and df.shape[1] == len(_cols_)
        except (AssertionError, FileNotFoundError):
            # print(query)
            pass
        if example_type == 'positive':
            df = df[df['ans_spans'] != 'N/A']
            assert df.shape[0] <= 10
        elif example_type == 'negative':
            df = df[df['ans_spans'] == 'N/A']
            assert df.shape[0] <= 10
        total += df.shape[0]

        for _, row in df.iterrows():
            pass_cnt = 0
            actual_spans = row['ans_spans']
            gen_spans = [row[x] for x in [f'ans_{i}' for i in range(n)] if row[x].strip()]
            bleu_gen_spans = [row[x] for x in [f'ans_{i}' for i in range(n)]]

            # bleu
            corpus_sentence_bleu += score_at_k(actual_spans, bleu_gen_spans, k)
            prev_pass_cnt = pass_cnt
            for spans in gen_spans:
                if len(actual_spans.split(':::-:::')) == 1:
                    pass_cnt += singlespan_eq(actual_spans, spans)
                else:
                    pass_cnt += multispan_eq(actual_spans, spans)
                if (k == 10 and example_type == 'negative' and pass_cnt > prev_pass_cnt
                        and 'Inconsistent' in query):
                    print(row['query'], row['file_path'])
                prev_pass_cnt = pass_cnt
            assert pass_cnt <= 10

            # calculate pass@k
            pass_at_k += cal_pass_at_k(n, pass_cnt, k)
        querywise_em[query] = (pass_at_k - prev_pass_at_k, df.shape[0])
    print(f"Correct with pass@{k}: {pass_at_k}/{total} = {pass_at_k/total}")

    # # bleu
    # print(len(references), len(candidates), total)
    # # assert len(references) == len(candidates) == total
    # corpus_bleu_score = corpus_bleu(references, candidates,
    #                                 weights=(0.25, 0.25, 0.25, 0.25),
    #                                 smoothing_function=smoothing_fn.method2)

    # print('Corpus BLEU: ', corpus_bleu_score / total)
    print('Sentence BLEU: ', corpus_sentence_bleu / total)
    return querywise_em


def get_combined_spans(gen_spans):
    bleu_gen_spans = []
    for span in gen_spans:
        temp_span = ''
        try:
            gen_ans_spans = span.split('Supporting fact span(s)\n```python')[0].strip().strip('```').strip()
            temp_span += gen_ans_spans
        except IndexError:
            pass

        try:
            gen_sf_spans = span.split('Supporting fact span(s)\n```python')[1].strip()
            if not temp_span:
                temp_span += gen_sf_spans
            else:
                temp_span += ' ' + gen_sf_spans
        except IndexError:
            pass

        bleu_gen_spans.append(temp_span)

    return bleu_gen_spans


def pass_at_k_with_sf(log_path, all_queries, query_folderName_map, k, n=10):
    pass_at_k = 0
    total = 0
    _cols_ = ['i', 'query', 'file_path', 'prompt', 'ans_spans', 'sf_spans']
    _cols_.extend([f'ans_{i}' for i in range(n)])
    querywise_em = {}
    corpus_sentence_bleu = 0
    corpus_sentence_bleu = 0
    for query in all_queries:
        prev_pass_at_k = pass_at_k
        try:
            df = pd.read_csv(__FILE_DIR__ / f'{log_path}/{query_folderName_map[query]}_logs.csv',
                             names=_cols_, keep_default_na=False)
            assert df.shape[0] <= 10 and df.shape[1] == len(_cols_)
        except (AssertionError, FileNotFoundError):
            # print(query)
            pass
        total += df.shape[0]

        for _, row in df.iterrows():
            pass_cnt = 0
            actual_ans_spans = row['ans_spans']
            actual_sf_spans = row['sf_spans']
            gen_spans = [row[x] for x in [f'ans_{i}' for i in range(n)] if row[x].strip()]

            # bleu
            bleu_actual_spans = ':::-:::'.join([actual_ans_spans, actual_sf_spans])
            bleu_gen_spans = get_combined_spans(gen_spans)
            corpus_sentence_bleu += score_at_k(bleu_actual_spans, bleu_gen_spans, k)

            prev_pass_cnt = pass_cnt
            for _, spans in enumerate(gen_spans):
                try:
                    gen_ans_spans = spans.split('Supporting fact span(s)\n```python')[0].strip().strip('```').strip()
                    gen_sf_spans = spans.split('Supporting fact span(s)\n```python')[1].strip()
                except IndexError:
                    # consider as correct spans not being produced
                    continue

                ans_em = False
                sf_em = False
                if len(actual_ans_spans.split(':::-:::')) == 1:
                    ans_em = (singlespan_eq(actual_ans_spans, gen_ans_spans) == 1)
                else:
                    ans_em = (multispan_eq(actual_ans_spans, gen_ans_spans) == 1)

                if len(actual_sf_spans.split(':::-:::')) == 1:
                    sf_em = (singlespan_eq(actual_sf_spans, gen_sf_spans) == 1)
                else:
                    sf_em = (multispan_eq(actual_sf_spans, gen_sf_spans) == 1)

                if ans_em and sf_em:
                    pass_cnt += 1
                # if (k == 10 and pass_cnt > prev_pass_cnt
                #         and 'Inconsistent' in query):
                #     print(row['query'], row['file_path'])
                prev_pass_cnt = pass_cnt
            assert pass_cnt <= 10

            # calculate pass@k
            pass_at_k += cal_pass_at_k(n, pass_cnt, k)
        querywise_em[query] = (pass_at_k - prev_pass_at_k, df.shape[0])
    print(f"Correct with pass@{k}: {pass_at_k}/{total} = {pass_at_k/total}")
    print('Sentence BLEU: ', corpus_sentence_bleu / total)

    return querywise_em


def eval(log_path, example_type, with_sf=False):
    with open(__FILE_DIR__ / 'resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)
    all_queries = list(query_folderName_map.keys())

    querywise_results = []
    for k_i in [1, 2, 5, 10]:
        if with_sf:
            querywise_em = pass_at_k_with_sf(log_path, all_queries, query_folderName_map, k=k_i)
            querywise_results.append(querywise_em)
        else:
            querywise_em = pass_at_k(log_path, all_queries, query_folderName_map, k=k_i, example_type=example_type)
            querywise_results.append(querywise_em)
    return querywise_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_folder", "-g", type=str, required=True)
    parser.add_argument("--with_sf", type=bool, default=False)
    parser.add_argument("--querywise", type=bool, default=False)

    nck_dict = build_nck()
    args = parser.parse_args()
    if args.with_sf:
        print('-' * 50, ' Positive ', '-' * 50)
        pos_querywise_results = eval(args.generated_folder, 'positive', args.with_sf)

        if args.querywise:
            querywise_csv_path = f"analyses/{args.generated_folder.split('/')[0]}"
            if not Path(querywise_csv_path).exists():
                os.makedirs(querywise_csv_path)
            rows = [['Query',
                     'Pos_@1', 'Pos_@2', 'Pos_@5', 'Pos_@10']]
            all_queries = list(pos_querywise_results[0].keys())
            for query in all_queries:
                row = [query]
                for split_results in [pos_querywise_results]:
                    for i, _ in enumerate([1, 2, 5, 10]):
                        row.append(round(split_results[i][query][0] * 100 / split_results[i][query][1], 2))
                rows.append(row)

            with open(f'{querywise_csv_path}/querywise_LLM_results.csv', "w") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(rows)
            print(f'Querywise results are written to {querywise_csv_path}/querywise_LLM_results.csv')
    else:
        print('-' * 50, ' All ', '-' * 50)
        all_querywise_results = eval(args.generated_folder, 'both')
        print('-' * 50, ' Positive ', '-' * 50)
        pos_querywise_results = eval(args.generated_folder, 'positive')
        print('-' * 50, ' Negative ', '-' * 50)
        neg_querywise_results = eval(args.generated_folder, 'negative')

        if args.querywise:
            querywise_csv_path = f"analyses/{args.generated_folder.split('/')[0]}"
            if not Path(querywise_csv_path).exists():
                os.makedirs(querywise_csv_path)
            rows = [['Query',
                     'All_@1', 'All_@2', 'All_@5', 'All_@10',
                     'Pos_@1', 'Pos_@2', 'Pos_@5', 'Pos_@10',
                     'Neg_@1', 'Neg_@2', 'Neg_@5', 'Neg_@10']]
            all_queries = list(all_querywise_results[0].keys())
            for query in all_queries:
                row = [query]
                for split_results in [all_querywise_results, pos_querywise_results, neg_querywise_results]:
                    for i, _ in enumerate([1, 2, 5, 10]):
                        row.append(round(split_results[i][query][0] * 100 / split_results[i][query][1], 2))
                rows.append(row)

            with open(f'{querywise_csv_path}/querywise_LLM_results.csv', "w") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(rows)
            print(f'Querywise results are written to {querywise_csv_path}/querywise_LLM_results.csv')
