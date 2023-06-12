### Steps for data preparation
1. Compile the proto definitions. For a quick reference, follow this [blog](https://www.freecodecamp.org/news/googles-protocol-buffers-in-python/).  
`protoc -I=. --python_out=. <proto_path>`
2. Run `download_and_serialize_dataset.sh`. This script downloads required files, serializes query and file data into proto format.
3. Run files from `data_preparation` in the following order. Input for each file can be checked from corresponding test files.  
    3.1. `run_create_query_result.py`  
    3.2. `run_create_tokenized_files_labels.py`  
    3.3. `run_create_blocks_labels_dataset.py`  
    3.4. `run_create_block_subtokens_labels.py`  
    3.5. `run_create_groupwise_prediction_dataset.py` / `run_create_relevance_prediction_examples.py` / `run_create_span_prediction_training_examples.py` for twostep/relevance prediction/ span prediction data
