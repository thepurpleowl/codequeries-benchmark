import sys
from absl import flags
from tree_sitter import Language, Parser
import dataset_with_context_pb2
from create_block_subtokens_labels import (create_cubert_subtokens_labels,
                                           create_codebert_subtokens_labels)

from graphcodebertutils import create_graphcodebert_dataflow_subtokens_labels

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_type",
    None,
    "'cubert' / 'codebert' / 'graphcodebert'"
)

flags.DEFINE_string(
    "ordering_of_blocks",
    None,
    "What ordering of blocks to follow?\
    ('default_proto'/'line_number'/'random')"
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "Path to Cubert vocabulary file."
)

flags.DEFINE_string(
    "tokenized_block_label_protobuf_file",
    None,
    "Path to tokenized block label protobuf file."
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


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    tokenized_block_query_labels_dataset = (dataset_with_context_pb2.
                                            TokenizedBlockQueryLabelsDataset())

    for i in range(1, FLAGS.number_of_previous_dataset_splits + 1):
        split_dataset = (dataset_with_context_pb2.TokenizedBlockQueryLabelsDataset())
        with open(FLAGS.tokenized_block_label_protobuf_file + str(i), "rb") as fd:
            split_dataset.ParseFromString(fd.read())
            # join
            tokenized_block_query_labels_dataset.tokenized_block_query_labels_item.extend(
                split_dataset.tokenized_block_query_labels_item
            )
            tokenized_block_query_labels_dataset.example_types.extend(
                split_dataset.example_types
            )

    if(FLAGS.model_type == 'cubert'):
        dataset = create_cubert_subtokens_labels(FLAGS.ordering_of_blocks,
                                                 tokenized_block_query_labels_dataset,
                                                 FLAGS.vocab_file)
    elif(FLAGS.model_type == 'codebert'):
        dataset = create_codebert_subtokens_labels(FLAGS.ordering_of_blocks,
                                                   tokenized_block_query_labels_dataset)
    elif(FLAGS.model_type == 'graphcodebert'):
        PY_LANGUAGE = Language("./my-languages.so", "python")
        tree_sitter_parser = Parser()
        tree_sitter_parser.set_language(PY_LANGUAGE)

        dataset = create_graphcodebert_dataflow_subtokens_labels(FLAGS.ordering_of_blocks,
                                                                 tokenized_block_query_labels_dataset,
                                                                 tree_sitter_parser)

    # split the data
    dataset_len = len(dataset.block_query_subtokens_labels_item)
    split_len = (dataset_len / FLAGS.number_of_dataset_splits)

    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        temp = dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset()

        lower = (i - 1) * split_len
        upper = (i) * split_len

        for j in range(int(lower), int(upper)):
            temp.block_query_subtokens_labels_item.append(
                dataset.block_query_subtokens_labels_item[j]
            )
            temp.example_types.append(
                dataset.example_types[j]
            )

        with open(FLAGS.save_dataset_location + str(i), "wb") as fd:
            fd.write(temp.SerializeToString())
