import sys
from absl import flags
import dataset_with_context_pb2
from create_query_result import create_query_result
from create_query_result import create_query_result_merged

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "raw_programs_protobuf_file",
    None,
    "Path to serialized raw programs protobuf file."
)

flags.DEFINE_string(
    "raw_queries_protobuf_file",
    None,
    "Path to serialized raw queries protobuf file."
)

flags.DEFINE_string(
    "results_file",
    None,
    "Path to CodeQL analysis results file."
)

flags.DEFINE_string(
    "save_dataset_location",
    None,
    "Path to store the merged query results dataset."
)

flags.DEFINE_string(
    "positive_or_negative_examples",
    None,
    "Are the examples positive or negative? ('positive'/'negative')"
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    raw_programs = dataset_with_context_pb2.RawProgramDataset()
    raw_queries = dataset_with_context_pb2.RawQueryList()

    with open(FLAGS.raw_programs_protobuf_file, "rb") as fd:
        raw_programs.ParseFromString(fd.read())

    with open(FLAGS.raw_queries_protobuf_file, "rb") as fd:
        raw_queries.ParseFromString(fd.read())

    dataset = create_query_result(
        raw_programs, raw_queries,
        FLAGS.results_file, FLAGS.positive_or_negative_examples)

    dataset_merged = create_query_result_merged(dataset)

    with open(FLAGS.save_dataset_location, "wb") as fd:
        fd.write(dataset_merged.SerializeToString())
