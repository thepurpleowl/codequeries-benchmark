# %%
from tqdm import tqdm
import datasets

examples_data = datasets.load_dataset("thepurpleowl/codequeries", "twostep", split=datasets.Split.VALIDATION)

# %%
# to get all blocks of a file, use indices of twostep_dict
# in similar fashion, one can aggregate spans.
twostep_dict = {}  # dict(query_name, code_file_path) = indices of code blocks
for i, example_instance in enumerate(tqdm(examples_data)):
    twostep_key = (example_instance['query_name'], example_instance['code_file_path'])
    if twostep_key not in twostep_dict:
        twostep_dict[twostep_key] = [i]
    else:
        twostep_dict[twostep_key].append(i)

print(len(twostep_dict.keys()))
# %%
