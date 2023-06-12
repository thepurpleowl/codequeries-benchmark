from absl import flags
import sys
import create_raw_programs_dataset
from collections import namedtuple

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "data_source",
    None,
    "Specify if the programs' dataset is taken from github or from some\
    other source. Write github or other."
)

flags.DEFINE_string(
    "repository_name",
    None,
    "If the programs' dataset is taken from some GitHub repository, provide\
    the name of the repository."
)

flags.DEFINE_string(
    "file_with_unique_paths",
    None,
    "Provide the path of the file which stores the of unique paths of the\
    program files if taken from GitHub. (blob)"
)

flags.DEFINE_string(
    "source_name",
    None,
    "If the programs dataset is taken from some other resource, provide\
    the name of the source."
)

flags.DEFINE_string(
    "split_name",
    None,
    "If the programs dataset is taken from some other resource, provide\
    the name of the split (TRAIN/TEST/VALIDATION/UNDEFINED)"
)

flags.DEFINE_string(
    "programs_file_path",
    None,
    "If the programs' dataset is taken from some other resource provide\
    the path of the file which contains the paths of the program files\
    relative to the topmost level."
)

flags.DEFINE_string(
    "dataset_programming_language",
    None,
    "Provide the programming langauge used in the programs in the dataset\
    files."
)

flags.DEFINE_string(
    "downloaded_dataset_location",
    None,
    "Provide the path to the folder where the data has been downloaded to."
)

flags.DEFINE_string(
    "save_dataset_location",
    None,
    "File path where raw programs' dataset should be serialized to."
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    File_Path_Content = namedtuple("File_Path_Content", "path content")

    if(FLAGS.data_source == "other"):
        files = []
        with open(FLAGS.programs_file_path, "r") as f:
            for line in f:
                line = line.strip()
                files.append(line)

        file_content = []
        for i in range(len(files)):
            with open(FLAGS.downloaded_dataset_location + "/" + files[i],
                      "r") as f:
                temp = File_Path_Content(files[i], f.read())
                file_content.append(temp)

        dataset = (
            create_raw_programs_dataset.CreateRawProgramsDatasetNonGithub(
                FLAGS.source_name,
                FLAGS.split_name,
                FLAGS.dataset_programming_language,
                file_content
            ))

    elif(FLAGS.data_source == "github"):
        file_paths = []
        with open(FLAGS.file_with_unique_paths, "r") as f:
            for line in f:
                line = line.strip()
                file_paths.append(line)

        file_content = []
        for i in range(len(file_paths)):
            with open(FLAGS.downloaded_dataset_location + "/" + file_paths[i],
                      "r") as f:
                temp = File_Path_Content(file_paths[i], f.read())
                file_content.append(temp)

        dataset = create_raw_programs_dataset.CreateRawProgramsDatasetGithub(
            FLAGS.repository_name,
            FLAGS.dataset_programming_language,
            file_content
        )

    with open(FLAGS.save_dataset_location, "wb") as fd:
        fd.write(dataset.SerializeToString())
