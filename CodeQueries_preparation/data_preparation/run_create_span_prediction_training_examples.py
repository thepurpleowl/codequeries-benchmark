import sys
from absl import flags
import dataset_with_context_pb2
from create_span_prediction_training_examples import create_span_prediction_training_examples

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
    "number_of_dataset_splits",
    20,
    "Into how many splits previous dataset was split."
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "Path to Cubert vocabulary file."
)

flags.DEFINE_string(
    "save_dataset_location",
    None,
    "Path to store the Cubert examples dataset."
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    dataset = (dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset())

    for i in range(1, FLAGS.number_of_dataset_splits + 1):
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

    examples_for_span_pred = create_span_prediction_training_examples(
        dataset, FLAGS.vocab_file, FLAGS.model_type
    )

    with open(FLAGS.save_dataset_location, "wb") as f:
        f.write(examples_for_span_pred.SerializeToString())
