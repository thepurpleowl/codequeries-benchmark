import sys
from absl import flags
import dataset_with_context_pb2
from create_groupwise_prediction_dataset import create_groupwise_prediction_dataset

PATH_PREFIX = "../code-cubert/data_preparation/"
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_type",
    None,
    "'cubert' / 'codebert'"
)

flags.DEFINE_string(
    "block_subtoken_label_protobuf_file",
    None,
    "Path to tokenized block subtoken label protobuf file."
)

flags.DEFINE_integer(
    "number_of_previous_dataset_splits",
    20,
    "In how many splits previous dataset was stored."
)

flags.DEFINE_integer(
    "number_of_dataset_splits",
    4,
    "In how many splits to save the dataset."
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "Path to cubert vocabulary file."
)

flags.DEFINE_string(
    "save_dataset_location",
    None,
    "Path to store the cubert/codebert examples dataset."
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    dataset = (dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset())

    for i in range(1, FLAGS.number_of_previous_dataset_splits + 1):
        split_dataset = (dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset())
        with open(FLAGS.block_subtoken_label_protobuf_file + str(i), "rb") as fd:
            split_dataset.ParseFromString(fd.read())
            # join
            dataset.block_query_subtokens_labels_item.extend(
                split_dataset.block_query_subtokens_labels_item
            )
            dataset.example_types.extend(
                split_dataset.example_types
            )

    examples_for_group_pred = create_groupwise_prediction_dataset(
        dataset, FLAGS.vocab_file, FLAGS.model_type
    )

    # split the data
    dataset_len = len(examples_for_group_pred.examples)
    split_len = (dataset_len / FLAGS.number_of_dataset_splits)

    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        temp = dataset_with_context_pb2.ExampleForGroupwisePredictionDataset()

        lower = (i - 1) * split_len
        upper = (i) * split_len

        for j in range(int(lower), int(upper)):
            temp.examples.append(
                examples_for_group_pred.examples[j]
            )

        with open(FLAGS.save_dataset_location + str(i), "wb") as fd:
            fd.write(temp.SerializeToString())
