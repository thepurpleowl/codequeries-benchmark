#%%
import pickle
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
import numpy as np
__FILE_DIR__ = Path(__file__).parent

def em_at_any(log_path, all_queries, query_folderName_map, n=10):
    cnt_total = 0
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

        total += df.shape[0]
        cnt = 0
        for _, row in df.iterrows():
            actual_span = row['ans_spans']
            gen_spans = [row[x] for x in [f'ans_{i}' for i in range(n)] if row[x].strip()]
            found = False
            for span in gen_spans:
                if actual_span.strip() == span.strip():
                    cnt += 1
                    cnt_total += 1
                    found = True
                else:
                    if actual_span:
                        match = SequenceMatcher(None, actual_span, span).find_longest_match()
                        if (match.size/len(actual_span) > 0.9 and match.size/len(span) > 0.9):
                            cnt += 1
                            cnt_total += 1
                            found = True

                if found == True:
                    break
            # if not found:
            #     print(query, row['file_path'])
                # print(actual_span, gen_spans)
    print(f"Correct with any: {cnt_total}/{total} = {cnt_total/total}")

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

def eval(log_path, example_type):
    n = 10
    with open(__FILE_DIR__ / 'resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)
    all_queries = list(query_folderName_map.keys())
    # all_queries = ['Comparison of constants']

    pass_at_k(log_path, all_queries, query_folderName_map, k=1, example_type=example_type)
    pass_at_k(log_path, all_queries, query_folderName_map, k=2, example_type=example_type)
    pass_at_k(log_path, all_queries, query_folderName_map, k=5, example_type=example_type)
    pass_at_k(log_path, all_queries, query_folderName_map, k=10, example_type=example_type)
    # em_at_any(log_path, all_queries, query_folderName_map)


if __name__ == "__main__":
    eval('test_dir_file_random_each/logs', 'both')
    print('-'*50)
    eval('test_dir_file_random_each/logs', 'positive')
    print('-'*50)
    eval('test_dir_file_random_each/logs', 'negative')
