import math
import pandas as pd
import dataset_with_context_pb2
import re
from collections import namedtuple


Span = namedtuple('Span', 'type start_line start_col end_line end_col')


def supporting_fact_overlaps(supporting_fact, result_span):
    """
    Whether supporting_fact overlaps result_span.
    Args:
        supporting_fact: supoortiung fact span
        result_span: result_span span
    Returns:
        True/False
    """
    if((supporting_fact.end_line >= result_span.start_line
            and supporting_fact.end_line <= result_span.end_line)
            or (supporting_fact.start_line >= result_span.start_line
                and supporting_fact.start_line <= result_span.end_line)):
        if(supporting_fact.end_line == result_span.start_line
                and supporting_fact.end_col < result_span.start_col):
            return False
        elif(supporting_fact.start_line == result_span.end_line
                and supporting_fact.start_col > result_span.end_col):
            return False
        else:
            return True

    return False


def get_result_locations(result_row, positive_or_negative_examples):
    """
    This function returns result location along with
    set of non-overlapping supporting facts.
    Args:
        result_row: specific row of CodeQL result csv.
    Returns:
        A set of non-overlapping supporting fact and spans.
    """
    supporting_facts = set()
    nonoverlapping_supporting_facts = []
    # Sometimes the results don't mention proper start/end line/
    # column. The following check helps avoid errors arising out of
    # this issue.
    if(math.isnan(result_row.Start_line) is True):
        return
    start_line = int(result_row.Start_line)
    if(math.isnan(result_row.End_line) is True):
        return
    end_line = int(result_row.End_line)
    if(math.isnan(result_row.Start_column) is True):
        return
    start_column = int(result_row.Start_column)
    if(math.isnan(result_row.End_column) is True):
        return
    end_column = int(result_row.End_column)

    # add the result span
    result_span = Span('RESULT', start_line, start_column, end_line, end_column)

    # get supporting fact spans
    if(positive_or_negative_examples == "positive"):
        matches = re.findall(r"relative:\/\/\/[a-zA-Z0-9_.]*:(\d+):(\d+):(\d+):(\d+)", result_row.Message)
    else:
        matches = []

    for match in matches:
        if(len(match) != 4):
            continue
        elif((int(match[2]) - int(match[0]) == 0) and (int(match[3]) - int(match[1]) == 0)):
            # to ignore built-in spans like 0:0:0:0
            continue
        try:
            supporting_fact = Span('SUPPORTING_FACT',
                                   int(match[0]),
                                   int(match[1]),
                                   int(match[2]),
                                   int(match[3]))
            supporting_facts.add(supporting_fact)
        except ValueError:
            # if regex capture something similar to span
            # but not a span
            continue
    # check overlap with result span
    for sf in supporting_facts:
        if not (supporting_fact_overlaps(sf, result_span)
                or supporting_fact_overlaps(result_span, sf)):
            nonoverlapping_supporting_facts.append(sf)

    result_location = dataset_with_context_pb2.ResultLocation()
    result_location.start_line = result_span.start_line
    result_location.end_line = result_span.end_line
    result_location.start_column = result_span.start_col
    result_location.end_column = result_span.end_col
    result_location.message = result_row.Message

    for span in nonoverlapping_supporting_facts:
        supporting_fact_location = dataset_with_context_pb2.SupportingFactLocation()

        supporting_fact_location.start_line = span.start_line
        supporting_fact_location.end_line = span.end_line
        supporting_fact_location.start_column = span.start_col
        supporting_fact_location.end_column = span.end_col

        result_location.supporting_fact_locations.append(supporting_fact_location)

    return result_location


def create_query_result(raw_programs_dataset, raw_query_dataset,
                        results_file, positive_or_negative_examples):
    """
    This function helps to ingest data into a protobuf to create dataset with
    raw program files, raw queries and corresponding results.
    Args:
        raw_programs_file: Serialized raw programs protobuf.
        raw_queries_file: Serialized raw queries protobuf.
        results_file: Path to file which contains CodeQL analysis results.
        positive_or_negative_examples: positive/negative
    Returns:
        A protobuf storing results from CodeQL analysis on a codebase and
        information about corresponding raw programs and queries.
    """
    # Reading the CodeQL analysis results, and defining names for the columns
    # in the results.
    results_file_df = pd.read_csv(results_file, names=[
                                  "Name", "Description", "Severity", "Message",
                                  "Path", "Start_line", "Start_column",
                                  "End_line", "End_column"])

    # Creating a dictionary for easy access to index of an item within raw
    # programs protobuf, given the path of the raw file.
    path_to_index = {}

    for i in range(len(raw_programs_dataset.raw_program_dataset)):
        path_to_index[raw_programs_dataset.raw_program_dataset[i]
                      .file_path.dataset_file_path.unique_file_path] = i

    # Creating a dictionary for easy access to index of an item within raw
    # queries protobuf, given the name of the query.
    name_to_index = {}

    for i in range(len(raw_query_dataset.raw_query_set)):
        name = raw_query_dataset.raw_query_set[i].metadata.name
        name_to_index[name] = i

    dataset = dataset_with_context_pb2.RawResultDataset()

    for i in range(len(results_file_df)):
        result_location = get_result_locations(results_file_df.iloc[i],
                                               positive_or_negative_examples)

        raw_query_result = dataset_with_context_pb2.RawQueryResult()
        raw_query_result.result_location.CopyFrom(result_location)

        path = results_file_df.Path[i][1:]
        index = path_to_index[path]
        raw_program = raw_programs_dataset.raw_program_dataset[index]
        raw_query_result.raw_file.CopyFrom(raw_program)

        name = results_file_df.Name[i]
        query_index = name_to_index[name]
        raw_query = raw_query_dataset.raw_query_set[query_index]
        raw_query_result.query.CopyFrom(raw_query)

        dataset.query_and_files_results.append(raw_query_result)

    return dataset


def create_query_result_merged(dataset_initial):
    """
    This function takes as input a protobuf which has a list of
    raw program files, raw query files and corresponding CodeQL
    analysis results. It then produces as output a protobuf which has
    as result a list of raw program files, raw query files and a list of
    results from CodeQL analysis, merged together.
    Args:
        dataset_initial: Output from the function "create_query_result".
    Returns:
        Raw merged results dataset.
    """

    # A dictionary to keep track of CodeQL analysis results where the raw
    # program file and raw query file are the same. It is used to merge them
    # later.
    trace_results = {}

    for i in range(len(dataset_initial.query_and_files_results)):
        key = dataset_initial.query_and_files_results[i].raw_file.file_path.\
            dataset_file_path.unique_file_path + \
            dataset_initial.query_and_files_results[i].query.query_path.unique_path
        if key in trace_results:
            trace_results[key].append(i)
        else:
            trace_results[key] = [i]

    merged_dataset = dataset_with_context_pb2.RawMergedResultDataset()

    # Merging the CodeQL results with the same raw program file and same raw
    # query file.
    for key in trace_results:
        merged_result_unit = dataset_with_context_pb2.RawMergedQueryResult()
        first_index = trace_results[key][0]
        merged_result_unit.raw_file_path.CopyFrom(
            dataset_initial.query_and_files_results[first_index].raw_file)
        merged_result_unit.query.CopyFrom(
            dataset_initial.query_and_files_results[first_index].query)
        merged_result_unit.resultlocation.append(
            dataset_initial.query_and_files_results[
                first_index].result_location)

        if(len(trace_results[key]) > 1):
            for j in range(1, len(trace_results[key])):
                next_index = trace_results[key][j]
                merged_result_unit.resultlocation.append(
                    dataset_initial.query_and_files_results[
                        next_index].result_location)

        assert len(merged_result_unit.resultlocation) == len(trace_results[key])

        merged_dataset.query_and_files_results.append(merged_result_unit)

    return merged_dataset
