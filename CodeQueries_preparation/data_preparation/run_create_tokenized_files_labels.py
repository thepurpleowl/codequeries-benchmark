import sys
from absl import flags
import dataset_with_context_pb2
from create_tokenized_files_labels import create_tokenized_files_labels

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "merged_query_result_protobuf_file",
    None,
    "Path to serialized query results protobuf file."
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
    "positive_or_negative_examples",
    None,
    "Are the examples positive or negative? ('positive'/'negative')"
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    dataset_merged = dataset_with_context_pb2.RawMergedResultDataset()

    with open(FLAGS.merged_query_result_protobuf_file, "rb") as fd:
        dataset_merged.ParseFromString(fd.read())

    tokenized_files_with_labels = create_tokenized_files_labels(
        dataset_merged, FLAGS.positive_or_negative_examples)

    # split the data
    dataset_len = len(tokenized_files_with_labels.tokens_and_labels)
    split_len = (dataset_len / FLAGS.number_of_dataset_splits)

    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        temp = dataset_with_context_pb2.TokenizedQueryProgramLabelsDataset()

        lower = (i - 1) * split_len
        upper = (i) * split_len

        for j in range(int(lower), int(upper)):
            temp.tokens_and_labels.append(
                tokenized_files_with_labels.tokens_and_labels[j]
            )
        temp.example_type = (dataset_with_context_pb2.
                             ExampleType.Value(FLAGS.positive_or_negative_examples))

        with open(FLAGS.save_dataset_location + str(i), "wb") as fd:
            fd.write(temp.SerializeToString())
