from absl import flags
import sys
import create_raw_codeql_queryset

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "labeled_queries_file_path",
    None,
    "Path of the file which contains the paths of the CodeQL query files \
    and manually labeled Hops and Span information."
)

flags.DEFINE_string(
    "target_programming_language",
    None,
    "Programming langauge for which the CodeQL query files can be used \
    to analyze the codebase."
)

flags.DEFINE_string(
    "github_auth",
    None,
    "Github API Basic Authentication to access blobs. \
    Value given as github_username:personal_access_token"
)

flags.DEFINE_string(
    "save_queryset_location",
    None,
    "File path where CodeQL query set should be serialized to."
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    create_raw_codeql_queryset.serialize_queryset(
        FLAGS.labeled_queries_file_path,
        FLAGS.target_programming_language,
        FLAGS.github_auth,
        FLAGS.save_queryset_location
    )
