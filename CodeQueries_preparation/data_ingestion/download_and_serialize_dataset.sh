wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz
wget https://github.com/google-research-datasets/eth_py150_open/raw/master/train__manifest.json
wget https://github.com/google-research-datasets/eth_py150_open/raw/master/eval__manifest.json
wget https://github.com/google-research-datasets/eth_py150_open/raw/master/dev__manifest.json

touch dev_program_files_paths.txt

jq -r ".[] | .filepath" dev__manifest.json | while read i; do
    echo $i >> dev_program_files_paths.txt
done

touch train_program_files_paths.txt

jq -r ".[] | .filepath" train__manifest.json | while read i; do
    echo $i >> train_program_files_paths.txt
done

touch eval_program_files_paths.txt

jq -r ".[] | .filepath" eval__manifest.json | while read i; do
    echo $i >> eval_program_files_paths.txt
done

tar -xvzf py150_files.tar.gz

tar -xvzf data.tar.gz

python run_create_raw_programs_dataset.py --data_source=other --source_name=eth_py150_open \
--split_name=TRAIN --programs_file_path=$(pwd)/train_program_files_paths.txt \
--dataset_programming_language=Python --downloaded_dataset_location=$(pwd)/data \
--save_dataset_location=$(pwd)/train_raw_programs_serialized

python run_create_raw_programs_dataset.py --data_source=other --source_name=eth_py150_open \
--split_name=VALIDATION --programs_file_path=$(pwd)/dev_program_files_paths.txt \
--dataset_programming_language=Python --downloaded_dataset_location=$(pwd)/data \
--save_dataset_location=$(pwd)/dev_raw_programs_serialized

python run_create_raw_programs_dataset.py --data_source=other --source_name=eth_py150_open \
--split_name=TEST --programs_file_path=$(pwd)/eval_program_files_paths.txt \
--dataset_programming_language=Python --downloaded_dataset_location=$(pwd)/data \
--save_dataset_location=$(pwd)/eval_raw_programs_serialized

python run_create_raw_codeql_queryset.py --labeled_queries_file_path=$(pwd)/$3 \
--target_programming_language=Python --github_auth=$1:$2 --save_queryset_location=$(pwd)/raw_queries_serialized

rm -rf dev_program_files_paths.txt train_program_files_paths.txt eval_program_files_paths.txt \
python50k_eval.txt python100k_train.txt \
dev__manifest.json data.tar.gz py150_files.tar.gz \
README.md github_repos.txt data train__manifest.json eval__manifest.json