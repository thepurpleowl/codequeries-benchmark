#%%
import pickle
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
__FILE_DIR__ = Path(__file__).parent


def cal_pass_at_k(n, c, k):
    if n -c <k:
        return 1
    return 1 - np.prod(1.0 - k/np.arange(n-c+1, n+1))

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

def pass_at_k(log_path, all_queries, query_folderName_map, k, example_type, n=10):
    pass_at_k = 0
    total = 0
    _cols_ = ['i', 'query', 'file_path', 'prompt', 'ans_spans']
    _cols_.extend([f'ans_{i}' for i in range(n)])
    for query in all_queries:
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

            for spans in gen_spans:
                if len(actual_spans.split(':::-:::')) == 1:
                    pass_cnt += singlespan_eq(actual_spans, spans)
                else:
                    pass_cnt += multispan_eq(actual_spans, spans)
            assert pass_cnt <= 10

            # calculate pass@k
            pass_at_k += cal_pass_at_k(n, pass_cnt, k)
    print(f"Correct with pass@{k}: {pass_at_k}/{total} = {pass_at_k/total}")

def pass_at_k_with_sf(log_path, all_queries, query_folderName_map, k, n=10):
    pass_at_k = 0
    total = 0
    _cols_ = ['i', 'query', 'file_path', 'prompt', 'ans_spans', 'sf_spans']
    _cols_.extend([f'ans_{i}' for i in range(n)])
    for query in all_queries:
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

            for _, spans in enumerate(gen_spans):
                try:
                    gen_ans_spans = spans.split('Supporting fact span(s)\n```python')[0].strip().strip('```').strip()
                    gen_sf_spans = spans.split('Supporting fact span(s)\n```python')[1].strip()
                except Exception as e:
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
            assert pass_cnt <= 10

            # calculate pass@k
            pass_at_k += cal_pass_at_k(n, pass_cnt, k)
    print(f"Correct with pass@{k}: {pass_at_k}/{total} = {pass_at_k/total}")

def eval(log_path, example_type, with_sf=False):
    n = 10
    with open(__FILE_DIR__ / 'resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)
    all_queries = list(query_folderName_map.keys())

    for k_i in [1, 2, 5, 10]:
        if with_sf:
            pass_at_k_with_sf(log_path, all_queries, query_folderName_map, k=k_i)
        else:
            pass_at_k(log_path, all_queries, query_folderName_map, k=k_i, example_type=example_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_folder", "-g", type=str, required=True)
    parser.add_argument("--with_sf", type=bool, default=False)

    args = parser.parse_args()
    if args.with_sf:
        eval(args.generated_folder, 'positive', args.with_sf)
    else:
        eval(args.generated_folder, 'both')
        print('-'*50)
        eval(args.generated_folder, 'positive')
        print('-'*50)
        eval(args.generated_folder, 'negative')
