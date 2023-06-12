import sys
from absl import flags
import dataset_with_context_pb2


FLAGS = flags.FLAGS


flags.DEFINE_string(
    "positive_examples_files",
    None,
    "Path to positive examples."
)

flags.DEFINE_string(
    "negative_examples_files",
    None,
    "Path to negative examples."
)

flags.DEFINE_integer(
    "number_of_dataset_splits",
    4,
    "In how many splits to save the dataset."
)

flags.DEFINE_string(
    "save_dataset_location",
    None,
    "Path to store the final block subtokens files."
)

if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    data = dataset_with_context_pb2.ExampleforSpanPredictionDataset()
    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        split_dataset = (dataset_with_context_pb2.ExampleforSpanPredictionDataset())
        with open(FLAGS.positive_examples_files + str(i), "rb") as fd:
            split_dataset.ParseFromString(fd.read())
            # join
            data.examples.extend(
                split_dataset.examples
            )
    for i in range(len(data.examples)):
        assert data.examples[i].example_type == 1
    print(len(data.examples))

    negative_data = dataset_with_context_pb2.ExampleforSpanPredictionDataset()
    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        split_dataset = (dataset_with_context_pb2.ExampleforSpanPredictionDataset())
        with open(FLAGS.negative_examples_files + str(i), "rb") as fd:
            split_dataset.ParseFromString(fd.read())
            # join
            negative_data.examples.extend(
                split_dataset.examples
            )
    for i in range(len(negative_data.examples)):
        assert negative_data.examples[i].example_type == 0
    print(len(negative_data.examples))

    data.examples.extend(negative_data.examples)
    print(len(data.examples))

    # split the data
    dataset_len = len(data.examples)
    split_len = (dataset_len / FLAGS.number_of_dataset_splits)

    for i in range(1, FLAGS.number_of_dataset_splits + 1):
        temp = dataset_with_context_pb2.ExampleforSpanPredictionDataset()

        lower = (i - 1) * split_len
        upper = (i) * split_len

        for j in range(int(lower), int(upper)):
            temp.examples.append(
                data.examples[j]
            )

        with open(FLAGS.save_dataset_location + str(i), "wb") as fd:
            fd.write(temp.SerializeToString())
