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

#%%
def count_file_tokens(file_path: str, model_name: str="gpt-35-turbo"):
    with open('/home/t-susahu/CodeQueries/data' + f'/{file_path}', 'r') as f:
       input_str = f.read()
    if model_name in TOKENIZER_MODEL_ALIAS_MAP:
        model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_str))
    return {'file_tokens': num_tokens}

def generate_querywise_test_samples():
  dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
  partitioned_data_all = {query['name']: dataset.map(lambda x: count_file_tokens(x['code_file_path'])).filter(lambda x: x["query_name"] == query['name'] and x["file_tokens"] <3000 ) for query in query_data}

  with open('resources/partitioned_data_all.pkl', 'wb') as f: 
      pickle.dump(partitioned_data_all, f)

def generate_querywise_train_all_samples():
  dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TRAIN)
  partitioned_data_train_all = {query['name']: dataset.filter(lambda x: x["query_name"] == query['name']) for query in query_data}

  with open('resources/partitioned_data_train_all.pkl', 'wb') as f: 
      pickle.dump(partitioned_data_train_all, f)

def generate_querywise_train_samples():
  dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TRAIN)
  partitioned_data_train_1000 = {query['name']: dataset.map(lambda x: count_file_tokens(x['code_file_path'])).filter(lambda x: x["query_name"] == query['name'] and x["file_tokens"] <1000 ) for query in query_data}

  with open('resources/partitioned_data_train_1000.pkl', 'wb') as f: 
      pickle.dump(partitioned_data_train_1000, f)

#%%
def get_querywise_test_stat():
  dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
  with open('resources/partitioned_data_all.pkl', 'rb') as f: 
    partitioned_data_all = pickle.load(f)

  total = 0
  for query in all_queries:
    total += partitioned_data_all[query].shape[0]
    all_files = dataset['code_file_path']
    for ff in partitioned_data_all[query]['code_file_path']:
       assert ff in all_files

  print(dataset.shape[0], total)
# get_querywise_test_stat()
#%%
def get_querywise_train_stat():
  dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TRAIN)
  with open('resources/partitioned_data_train_all.pkl', 'rb') as f: 
    partitioned_data_train_1000 = pickle.load(f)

  total = 0
  for query in all_queries:
    total += partitioned_data_train_1000[query].shape[0]
    all_files = dataset['code_file_path']
    for ff in partitioned_data_train_1000[query]['code_file_path']:
       assert ff in all_files

  print(dataset.shape[0], total)
get_querywise_train_stat()
# %%
def sample_data(query_data, s=10):
    pos_files = set(query_data.filter(lambda x: x["example_type"] == 1)['code_file_path'])
    neg_files = set(query_data.filter(lambda x: x["example_type"] == 0)['code_file_path'])
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

def get_sampled_file_metadata(partitioned_data_path, output_path):
  with open(partitioned_data_path, 'rb') as f: 
    partitioned_data = pickle.load(f)

  sampled_partitioned_data = {}
  for query in tqdm(all_queries):
    sampled_partitioned_data[query] = sample_data(partitioned_data[query])

  with open(output_path, 'wb') as f: 
      pickle.dump(sampled_partitioned_data, f)

def get_instances_for_files(sampled_data_path, dataset_setting, split):
   # dataset_setting: ideal/twostep
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
  get_sampled_file_metadata('resources/partitioned_data_all.pkl', 'resources/sampled_test_data.pkl')
  get_sampled_file_metadata('resources/partitioned_data_train_all.pkl', 'resources/sampled_train_all_data.pkl')
