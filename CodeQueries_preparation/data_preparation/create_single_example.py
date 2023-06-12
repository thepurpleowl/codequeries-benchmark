import sys
from absl import flags
from importlib import import_module
from subprocess import call

sys.path.insert(0, "../data_ingestion/")
raw_codeql_queryset_pb2 = import_module('raw_codeql_queryset_pb2')

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "save_location",
    None,
    "Path to store intermediate data and example"
)

flags.DEFINE_string(
    "file_with_path",
    None,
    "Provide the path of the file which stores the of\
    paths of the program files"
)

flags.DEFINE_string(
    "raw_queries_protobuf_file",
    None,
    "Path to serialized raw queries protobuf file."
)

flags.DEFINE_string(
    "data_path",
    None,
    "Path to all examples"
)

flags.DEFINE_string(
    "result_file",
    None,
    "Path to CodeQL result csv"
)

flags.DEFINE_string(
    "example_type",
    None,
    "positive/negative"
)

flags.DEFINE_string(
    "aux_result_path",
    None,
    "positive/negative"
)

flags.DEFINE_string(
    "model_type",
    'cubert',
    "cubert/codebert"
)

flags.DEFINE_string(
    "block_ordering",
    'line_number',
    "line_number/random/--"
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "Cubert vocab file path, not reqd when model_type is CodeBERT"
)

if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    save_location = FLAGS.save_location
    result_file = FLAGS.result_file
    eg_type = FLAGS.example_type
    aux_result_path = FLAGS.aux_result_path
    model_type = FLAGS.model_type
    block_ordering = FLAGS.block_ordering
    vocab_file = FLAGS.vocab_file
    data_path = FLAGS.data_path
    path_file = FLAGS.file_with_path
    serialized_queries_path = FLAGS.raw_queries_protobuf_file

    serialized_src_file_path = save_location + '/source_file_serialized'

    call("python ../data_ingestion/run_create_raw_programs_dataset.py \
         --data_source=other --source_name=pyeth150_open --split_name=TEST \
         --dataset_programming_language=Python --programs_file_path=" + path_file
         + " --downloaded_dataset_location=" + data_path
         + " --save_dataset_location=" + serialized_src_file_path, shell=True)

    merged_query_results = save_location + '/merged_query_result'
    call("python run_create_query_result.py --raw_programs_protobuf_file="
         + serialized_src_file_path + " --raw_queries_protobuf_file=" + serialized_queries_path
         + " --results_file=" + result_file + " --save_dataset_location=" + merged_query_results, shell=True)

    tokenized_src_file = save_location + '/tokenized_src_file'
    call("python run_create_tokenized_files_labels.py --merged_query_result_protobuf_file="
         + merged_query_results + " --save_dataset_location=" + tokenized_src_file
         + " --positive_or_negative_examples=" + eg_type
         + " --number_of_dataset_splits=1", shell=True)

    tokenized_block_label = save_location + '/tokenized_block_label'
    call("python run_create_blocks_labels_dataset.py --tokenized_file_protobuf_file="
         + tokenized_src_file + " --save_dataset_location=" + tokenized_block_label
         + " --aux_result_path=" + aux_result_path
         + " --number_of_previous_dataset_splits=1 --number_of_dataset_splits=1", shell=True)

    block_subtoken_labels = save_location + '/block_subtoken_labels'
    call("python run_create_block_subtokens_labels.py --model_type="
         + model_type + " --ordering_of_blocks=" + block_ordering + " --vocab_file=" + vocab_file
         + " --tokenized_block_label_protobuf_file=" + tokenized_block_label
         + " --save_dataset_location=" + block_subtoken_labels
         + " --number_of_previous_dataset_splits=1 --number_of_dataset_splits=1", shell=True)

    model_example = save_location + '/model_example'
    call("python run_create_single_model_baseline_examples.py --model_type="
         + model_type + " --block_subtoken_label_protobuf_file=" + block_subtoken_labels
         + " --vocab_file=" + vocab_file + " --save_dataset_location=" + model_example
         + " --number_of_dataset_splits=1", shell=True)
