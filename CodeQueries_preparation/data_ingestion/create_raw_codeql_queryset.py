import raw_codeql_queryset_pb2 as raw_codeql_queryset_pb2
import csv
import hashlib
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
import base64

# To denote non-existance of a specific metadata information
MISSING_VALUE = "NA"
NON_DISTRIBUTABLE_QUERIES = ['Unused import',
                             '`__eq__` not overridden when adding attributes',
                             'Use of the return value of a procedure',
                             'Wrong number of arguments in a call',
                             'Comparison using is when operands support `__eq__`',
                             'Non-callable called',
                             '`__init__` method calls overridden method',
                             'Signature mismatch in overriding method',
                             'Conflicting attributes in base classes',
                             'Inconsistent equality and hashing',
                             'Flask app is run in debug mode',
                             'Wrong number of arguments in a class instantiation',
                             'Incomplete ordering',
                             'Missing call to `__init__` during object initialization',
                             '`__iter__` method returns a non-iterator']


def serialize_queryset(labeled_queries_file_path: str,
                       target_programming_language: str,
                       github_auth: str,
                       save_queryset_location: str):
    """
    This is the helper function which uses command line arguments to build
    CodeQL queryset protobufs.
    Args:
        labeled_queries_file_path:
            Path of the file which contains the paths of the CodeQL query files
            and manually labeled Hops and Span information.

        target_programming_language:
            Programming langauge for which the CodeQL query files can be used
            to analyze the codebase.

        github_auth:
            Github API Basic Authentication to access blobs.

        save_queryset_location
            File path where CodeQL query set should be serialized to.
    Returns:
        NA
        Store serialized CodeQL queryset on disk.

    """
    query_set_content = get_queryset_content(
        labeled_queries_file_path,
        github_auth
    )

    query_set = create_codeql_queryset(
        target_programming_language,
        query_set_content
    )

    with open(save_queryset_location, "wb") as fd:
        fd.write(query_set.SerializeToString())


def get_queryset_content(labeled_queries_file_path: str,
                         github_auth: str):
    """
    This function is used to get content of the query files.
    Args:
        labeled_queries_file_path:
            Path of the file which contains the paths of the CodeQL query files
            and manually labeled Hops and Span information.

        github_auth:
            Github API Basic Authentication to access blobs.
    Returns:
        A name-value pair(nvp)/dict object with parsed set of CodeQL query file
        content and manually labeled hops and span information.

    """
    query_set_content = []
    with open(labeled_queries_file_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        # Getting metadata and manual labels as name-value pair(nvp).
        for row in reader:
            query_metadata_nvp = dict()
            query_metadata_nvp["path"] = row["Path"]
            query_metadata_nvp["repo"] = ""
            query_metadata_nvp["url"] = row["Url"]
            query_metadata_nvp["hops"] = (
                "single_hop" if row["Hops"] == "1"
                else "multiple_hops" if row["Hops"] == "k"
                else "undefined_hop"
            )
            query_metadata_nvp["span"] = (
                "single_span" if row["Span"] == "1"
                else "multiple_spans" if row["Span"] == "k"
                else "undefined_span"
            )
            query_set_content.append(query_metadata_nvp)

    github_adapter = HTTPAdapter(max_retries=3)
    github_username = github_auth.split(":")[0]
    github_token = github_auth.split(":")[1]
    with requests.Session() as session:
        session.auth = (github_username, github_token)
        # Use `github_adapter` for all requests to endpoints that start with
        # this URL
        session.mount("https://api.github.com", github_adapter)

        try:
            # Access CodeQL query file content.
            for i in range(len(query_set_content)):
                github_url_split = query_set_content[i]["url"].split("/")
                username_repo = github_url_split[3] + "/" + github_url_split[4]
                relative_file_path = query_set_content[i]["path"]
                api_endpoint = "https://api.github.com/repos/" + \
                    username_repo + "/contents/" + relative_file_path

                response = session.get(api_endpoint)
                response_data = response.json()
                coded_string = response_data["content"]
                query_set_content[i]["content"] = base64.b64decode(
                    coded_string).decode("utf-8")

                query_set_content[i]["repo"] = "https://github.com/" + \
                    username_repo
                query_set_content[i]["url"] = response_data["git_url"]
        except ConnectionError as ce:
            print(ce)

    return query_set_content


def create_codeql_queryset(target_programming_language: str,
                           codeql_query_files: list):
    """
    This function helps ingest data into a protobuf, to create a set of
    CodeQL queries intended for a specific programming language. The queries
    will be used in RawQueryResult.
    Args:
        target_programming_language:
            Here we mention the target programming language for
            which the CodeQL query files are written.

        codeql_query_files:
            This is a list of dictionaries. Each element contains
            some metadata such as Hops and Span along with path
            to a CodeQL query file,and the corresponding content of
            the file.
    Returns:
        A protobuf containing the set of CodeQL queries along with metadata.

    """

    # Protobuf for the entire CodeQL query set.
    codeql_query_set = raw_codeql_queryset_pb2.RawQueryList()
    for query_nvp in codeql_query_files:
        # Protobuf for Github repo and repo relative file paths.
        file_path_on_github = raw_codeql_queryset_pb2.GitHubFilePath()
        file_path_on_github.repo = query_nvp["repo"]
        file_path_on_github.unique_path = query_nvp["url"]

        # Protobuf for metadata of a specific query file.
        query_metadata = raw_codeql_queryset_pb2.QueryMetadata()
        query_file_content = query_nvp["content"]
        metadata_nvp, full_metadata = parse_metadata(
            query_file_content,
            target_programming_language)

        query_metadata.name = (
            MISSING_VALUE if metadata_nvp.get("name") is None
            else metadata_nvp.get("name")
        )
        query_metadata.description = (
            MISSING_VALUE if metadata_nvp.get("description") is None
            else metadata_nvp.get("description")
        )
        query_metadata.severity = (
            MISSING_VALUE if metadata_nvp.get("problem.severity") is None
            else metadata_nvp.get("problem.severity")
        )
        query_metadata.message = (
            MISSING_VALUE if metadata_nvp.get("message") is None
            else metadata_nvp.get("message")
        )
        query_metadata.full_metadata = full_metadata

        # Protobuf for a specific query file.
        query = raw_codeql_queryset_pb2.Query()
        query.query_path.CopyFrom(file_path_on_github)
        # path used for queryID which is of type bytes, encode(): TBC
        query.queryID = hashlib.md5(
            (query.query_path.unique_path + query_file_content).encode("utf-8")
        ).digest()
        query.content = query_file_content
        query.metadata.CopyFrom(query_metadata)
        query.language = raw_codeql_queryset_pb2.Languages.Value(
            target_programming_language)
        query.hops = raw_codeql_queryset_pb2.Hops.Value(query_nvp["hops"])
        query.span = raw_codeql_queryset_pb2.Span.Value(query_nvp["span"])
        query.distributable = str(query.metadata.name).strip() not in NON_DISTRIBUTABLE_QUERIES

        codeql_query_set.raw_query_set.append(query)

    return codeql_query_set


def parse_metadata(query_file_content, target_programming_language):
    """
    This function parses the CodeQL query file to get specific metadata
    information about a specific CodeQL query file.
    Args:
        query_file_content:
                      This is the content of a CodeQL query file,
                      which will be parsed for metadata.
        target_programming_language:
            Here we mention the target programming language for
            which the CodeQL query files are written.
    Returns:
        A name-value pair(nvp)/dict object with parsed set of CodeQL query file
        metadata.

    """
    metadata_nvp = {}
    metadata_delimiter = "import " + target_programming_language.lower()

    # all metadata
    full_metadata = query_file_content.split(metadata_delimiter)[0].strip()
    # metadata after removing - spaces, /, *
    formatted_full_metadata = full_metadata.strip("/").split('\n')
    formatted_full_metadata = " ".join(
        [metadata.strip().lstrip('*') for metadata in formatted_full_metadata])

    # key specific formatting
    split_metadata = formatted_full_metadata.split("@")[1:]
    for metadata in split_metadata:
        space_removed_metadata = ' '.join(metadata.split())
        key = space_removed_metadata.split(" ")[0].strip()
        metadata_nvp[key] = space_removed_metadata[len(key):].strip()

    return metadata_nvp, full_metadata
