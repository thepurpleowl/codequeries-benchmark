import sys
import pandas as pd
from absl import flags
from tree_sitter import Language, Parser
import dataset_with_context_pb2
from create_blocks_labels_dataset import create_blocks_labels_dataset
from create_blocks_relevance_labels_dataset import create_blocks_relevance_labels_dataset
from contexts.get_context import __columns__

PATH_PREFIX = "../code-cubert/data_preparation/"
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tokenized_file_protobuf_file",
    None,
    "Path to tokenized source file protobuf file."
)

flags.DEFINE_integer(
    "number_of_previous_dataset_splits",
    20,
    "In how many splits previous dataset was stored."
)

flags.DEFINE_integer(
    "number_of_dataset_splits",
    20,
    "In how many splits to save the dataset."
)

flags.DEFINE_string(
    "save_dataset_location",
    None,
    "Path to store the Cubert examples dataset."
)

flags.DEFINE_string(
    "aux_result_path",
    None,
    "Path to store the Cubert examples dataset."
)

flags.DEFINE_string(
    "with_header",
    'yes',
    "yes/no"
)

flags.DEFINE_string(
    "with_simplified_relevance",
    None,
    "yes/no"
)

flags.DEFINE_string(
    "only_relevant_blocks",
    None,
    "yes/no - whether to get only relevant blocks or all blocks"
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    PY_LANGUAGE = Language("./my-languages.so", "python")
    # PY_LANGUAGE = Language(PATH_PREFIX + "my-languages.so", "python")

    tree_sitter_parser = Parser()
    tree_sitter_parser.set_language(PY_LANGUAGE)

    tokenized_files_with_labels = dataset_with_context_pb2.TokenizedQueryProgramLabelsDataset()
    for i in range(1, FLAGS.number_of_previous_dataset_splits + 1):
        split_dataset = (dataset_with_context_pb2.TokenizedQueryProgramLabelsDataset())
        with open(FLAGS.tokenized_file_protobuf_file + str(i), "rb") as fd:
            split_dataset.ParseFromString(fd.read())
            # join
            tokenized_files_with_labels.tokens_and_labels.extend(
                split_dataset.tokens_and_labels
            )
            if(i == 1):
                tokenized_files_with_labels.example_type = split_dataset.example_type
            else:
                assert tokenized_files_with_labels.example_type == split_dataset.example_type

    keep_header = (FLAGS.with_header == 'yes')
    aux_result_df = pd.read_csv(FLAGS.aux_result_path,
                                names=__columns__)
    if(FLAGS.with_simplified_relevance == 'yes'):
        tokenized_block_query_labels_dataset = create_blocks_relevance_labels_dataset(
            tokenized_files_with_labels, tree_sitter_parser,
            keep_header, aux_result_df, FLAGS.only_relevant_blocks
        )
    else:
        tokenized_block_query_labels_dataset = create_blocks_labels_dataset(
            tokenized_files_with_labels, tree_sitter_parser,
            keep_header, aux_result_df
        )

    # split the data
    dataset_len = len(tokenized_block_query_labels_dataset.tokenized_block_query_labels_item)
    split_len = (dataset_len / FLAGS.number_of_dataset_splits)

    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        temp = dataset_with_context_pb2.TokenizedBlockQueryLabelsDataset()

        lower = (i - 1) * split_len
        upper = (i) * split_len

        for j in range(int(lower), int(upper)):
            temp.tokenized_block_query_labels_item.append(
                tokenized_block_query_labels_dataset.tokenized_block_query_labels_item[j]
            )
            temp.example_types.append(
                tokenized_block_query_labels_dataset.example_types[j]
            )

        with open(FLAGS.save_dataset_location + str(i), "wb") as fd:
            fd.write(temp.SerializeToString())
