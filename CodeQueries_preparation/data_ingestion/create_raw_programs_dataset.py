import raw_programs_dataset_pb2 as raw_programs_dataset_pb2


def CreateRawProgramsDatasetGithub(repo_name: str,
                                   program_language: str,
                                   program_path_and_content: list):
    """
    This function helps ingest data into a protobuf, to creating raw
    programs dataset. This dataset will eventually be used to run
    CodeQL queries on, and to get answers to those queries. This function
    is used when the source of the programs' dataset is GitHub.
    Args:
        repo_name: When the programs are taken from GitHub, we need
                   to store the name of the repository here, as a string.
        program_language: Here we mention the programming language, in which
                      the files in this dataset are written.
        program_path_and_content: This is a list of tuples. Each tuple contains
                        the path to a program file, and the corresponding
                        content of the file. Both stored as strings.
    Returns:
        A protobuf containing the entire programs dataset, containing all
        kinds of information about the dataset.

    """
    # Protobuf for the entire dataset.
    entire_programs_dataset = raw_programs_dataset_pb2.RawProgramDataset()
    for i in range(len(program_path_and_content)):
        file_path_in_dataset = raw_programs_dataset_pb2.GitHubFilePath()

        file_path_in_dataset.repo = repo_name
        # This must be a unique path to the GitHub file. (blob)
        file_path_in_dataset.unique_path = program_path_and_content[i].path

        path_of_file = raw_programs_dataset_pb2.FilePath()
        path_of_file.dataset_file_path.CopyFrom(file_path_in_dataset)

        raw_program_file = raw_programs_dataset_pb2.RawProgramFile()
        raw_program_file.file_path.CopyFrom(path_of_file)
        raw_program_file.language = raw_programs_dataset_pb2.Languages.Value(
            program_language)

        # Storing the program contents.
        raw_program_file.file_content = program_path_and_content[i].content

        # Storing the entire dataset.
        entire_programs_dataset.raw_program_dataset.append(raw_program_file)

    return entire_programs_dataset


def CreateRawProgramsDatasetNonGithub(dataset_name: str, split_name: str,
                                      program_language: str,
                                      program_content_files: list
                                      ):
    """
    This function helps ingest data into a protobuf, to creating raw
    programs dataset. This dataset will eventually be used to run CodeQL
    queries on, and to get answers to those queries. This function is
    used when the source of the dataset is some place other than GitHub.
    Args:
        dataset_name: When the programs are taken from some place other than
                    GitHub, we need to store the name of the dataset here,
                    as a string.
        split_name: If the creators of the dataset have split the dataset
                    into train/test/validation, then we pass this information
                    here as a string.
        program_language: Here we mention the programming language, in which
                    the files in this dataset are written.
        program_content_files: This is a list of tuples. Each tuple contains
                    the path to a program file,and the corresponding
                    content of the file. Both stored as strings.
    Returns:
        A protobuf containing the entire programs dataset, containing
        all kinds of information about the dataset.

    """
    # Protobuf for the entire dataset.
    entire_programs_dataset = raw_programs_dataset_pb2.RawProgramDataset()
    for i in range(len(program_content_files)):
        # Protobuf to store information about program file paths.
        file_path_in_dataset = raw_programs_dataset_pb2.DatasetFilePath()
        file_path_in_dataset.source_name = dataset_name
        file_path_in_dataset.split = (
            raw_programs_dataset_pb2.datasetsplit.Value(
                split_name))
        file_path_in_dataset.unique_file_path = program_content_files[i].path

        path_of_file = raw_programs_dataset_pb2.FilePath()
        path_of_file.dataset_file_path.CopyFrom(file_path_in_dataset)

        raw_program_file = raw_programs_dataset_pb2.RawProgramFile()
        raw_program_file.file_path.CopyFrom(path_of_file)

        # Storing programming language information.
        raw_program_file.language = raw_programs_dataset_pb2.Languages.Value(
            program_language)

        # Storing the program content.
        raw_program_file.file_content = program_content_files[i].content

        entire_programs_dataset.raw_program_dataset.append(raw_program_file)

    return entire_programs_dataset
