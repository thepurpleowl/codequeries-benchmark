import dataset_with_context_pb2
from tqdm import tqdm
from collections import namedtuple
from dataclasses import dataclass
import sys
sys.path.insert(0, "..")
from cubert import python_tokenizer

Span = namedtuple('Span', 'span_type start_line start_col end_line end_col')


@dataclass
class LabelSpan:
    label: str
    span_start: str


def get_tokenized_stub_content(stub_block):
    """
    This function tokenizes program files and creates labels.
    Args:
        stub_block: Block with stub content
    Returns:
        Stub Block metadata to be used in create_blocks_label
    """
    # for both CuBERT/CodeBERT, starting tokenizer is same
    initial_tokenizer = python_tokenizer.PythonTokenizer()

    program_tokens_and_metadata = initial_tokenizer.tokenize_and_abstract(
        stub_block.content)

    stub_token_metadata = []
    for j in range(len(program_tokens_and_metadata)):
        tokens_labels_and_metadata = (
            dataset_with_context_pb2.TokensLabelsAndMetaData())

        # line and column field values are inconsequential as every token will be added
        tokens_labels_and_metadata.start_line = (stub_block.start_line)
        tokens_labels_and_metadata.end_line = (stub_block.end_line)
        tokens_labels_and_metadata.start_column = (stub_block.start_line)
        tokens_labels_and_metadata.end_column = (stub_block.end_line)
        tokens_labels_and_metadata.program_token = (
            program_tokens_and_metadata[j].spelling)
        tokens_labels_and_metadata.label = (
            dataset_with_context_pb2.OutputLabels.Value("O"))

        stub_token_metadata.append(tokens_labels_and_metadata)

    return stub_token_metadata


def get_tokenized_class_header(class_header):
    """
    This function tokenizes additional class header for each CLASS_FUNCTION block.
    Args:
        class_header: class header information of CLASS_FUNCTION/CLASS_OTHER block
    Returns:
        class_header tokens that to be appended with CLASS_FUNCTION block tokens
        in create_blocks_label
    """
    # for both CuBERT/CodeBERT, starting tokenizer is same
    initial_tokenizer = python_tokenizer.PythonTokenizer()

    program_tokens_and_metadata = initial_tokenizer.tokenize_and_abstract(
        class_header)

    class_header_metadata = []
    for j in range(len(program_tokens_and_metadata)):
        tokens_labels_and_metadata = (
            dataset_with_context_pb2.TokensLabelsAndMetaData())

        # line and column field values are inconsequential as every token will be added
        tokens_labels_and_metadata.start_line = -1
        tokens_labels_and_metadata.end_line = -1
        tokens_labels_and_metadata.start_column = -1
        tokens_labels_and_metadata.end_column = -1
        tokens_labels_and_metadata.program_token = (
            program_tokens_and_metadata[j].spelling)
        tokens_labels_and_metadata.label = (
            dataset_with_context_pb2.OutputLabels.Value("O"))

        class_header_metadata.append(tokens_labels_and_metadata)

    return class_header_metadata


def get_tokenized_perturbed_code(perturbed_code, line_number):
    """
    This function tokenizes perturbed_code.
    Args:
        perturbed_code: perturbed line of code
        line_number: perturbed line number
    Returns:
        Tokemized perturbed line of code with metadata
    """
    # for both CuBERT/CodeBERT, starting tokenizer is same
    initial_tokenizer = python_tokenizer.PythonTokenizer()

    program_tokens_and_metadata = initial_tokenizer.tokenize_and_abstract(perturbed_code)

    if(len(program_tokens_and_metadata) == 1
            and program_tokens_and_metadata[0].spelling == '___ERROR___'):
        raise Exception('Perturbed PL Tokenization Error')

    perturbed_code_metadata = []
    current_start_column = 0
    current_end_column = 0
    for j in range(len(program_tokens_and_metadata)):
        tokens_labels_and_metadata = (
            dataset_with_context_pb2.TokensLabelsAndMetaData())

        tokens_labels_and_metadata.program_token = (
            program_tokens_and_metadata[j].spelling)
        tokens_labels_and_metadata.label = (
            dataset_with_context_pb2.OutputLabels.Value("O"))
        tokens_labels_and_metadata.start_line = line_number
        tokens_labels_and_metadata.end_line = line_number
        # dummy column values for sorting later
        tokens_labels_and_metadata.start_column = current_start_column
        current_end_column = current_start_column + len(program_tokens_and_metadata[j].spelling)
        tokens_labels_and_metadata.end_column = current_end_column - 1
        current_start_column = current_end_column + 1

        perturbed_code_metadata.append(tokens_labels_and_metadata)

    return perturbed_code_metadata


def get_tokenized_query_name(query_name):
    """
    This function tokenizes query name, generally used to tokenize augmented query name.
    Args:
        query_name: augmented query name
    Returns:
        query_name tokens
    """
    initial_tokenizer = python_tokenizer.PythonTokenizer()
    query_tokens_and_metadata = initial_tokenizer.tokenize_and_abstract(query_name)[:-1]

    if(len(query_tokens_and_metadata) == 1
            and query_tokens_and_metadata[0].spelling == '___ERROR___'):
        raise Exception('Augmented NL Tokenization Error')

    return [token.spelling for token in query_tokens_and_metadata]


def supporting_fact_overlaps(supporting_fact, added_span):
    """
    Whether supporting_fact overlaps added_span.
    Args:
        supporting_fact: supoortiung fact span
        added_span: added_span span
    Returns:
        True/False
    """
    if((supporting_fact.end_line >= added_span.start_line
            and supporting_fact.end_line <= added_span.end_line)
            or (supporting_fact.start_line >= added_span.start_line
                and supporting_fact.start_line <= added_span.end_line)):
        if(supporting_fact.end_line == added_span.start_line
                and supporting_fact.end_col < added_span.start_col):
            return False
        elif(supporting_fact.start_line == added_span.end_line
                and supporting_fact.start_col > added_span.end_col):
            return False
        else:
            return True

    return False


def create_tokenized_files_labels(raw_merged_query_result_dataset,
                                  positive_or_negative_examples):
    """
    This function tokenizes program files and creates labels.
    Args:
        raw_merged_query_result_dataset: RawMergedResultDataset protobuf
        positive_or_negative_examples: positive/negative
    Returns:
        TokenizedQueryProgramLabelsDataset protobuf
    """

    # for both CuBERT/CodeBERT, starting tokenizer is same
    initial_tokenizer = python_tokenizer.PythonTokenizer()

    tokenized_program_query_labels_dataset = (dataset_with_context_pb2.
                                              TokenizedQueryProgramLabelsDataset())

    # engulfed_spans = 0
    missed_spans = 0  # possible wrong spans
    for i in tqdm(range(len(raw_merged_query_result_dataset.
                            query_and_files_results)), desc="Tokenized_files_labels"):

        tokenized_program_query_labels = (dataset_with_context_pb2.
                                          TokenizedProgramQueryLabels())

        tokenized_program_query_labels.query_and_files_results.CopyFrom(
            raw_merged_query_result_dataset.query_and_files_results[i]
        )

        program_content = (raw_merged_query_result_dataset.query_and_files_results[i].
                           raw_file_path.file_content)

        query_name = (raw_merged_query_result_dataset.
                      query_and_files_results[i].query.metadata.name)

        results = (raw_merged_query_result_dataset.query_and_files_results[i].
                   resultlocation)

        program_tokens_and_metadata = initial_tokenizer.tokenize_and_abstract(
            program_content)[:-1]

        query_tokens_and_metadata = initial_tokenizer.tokenize_and_abstract(
            query_name)[:-1]

        for j in range(len(query_tokens_and_metadata)):
            tokenized_program_query_labels.query_name_tokens.append(
                query_tokens_and_metadata[j].spelling
            )

        for j in range(len(program_tokens_and_metadata)):
            tokens_labels_and_metadata = (dataset_with_context_pb2.
                                          TokensLabelsAndMetaData())

            tokens_labels_and_metadata.start_line = (
                program_tokens_and_metadata[j].metadata.start.line)
            tokens_labels_and_metadata.end_line = (
                program_tokens_and_metadata[j].metadata.end.line)
            tokens_labels_and_metadata.start_column = (
                program_tokens_and_metadata[j].metadata.start.column)
            tokens_labels_and_metadata.end_column = (
                program_tokens_and_metadata[j].metadata.end.column)
            tokens_labels_and_metadata.program_token = (
                program_tokens_and_metadata[j].spelling)

            tokenized_program_query_labels.tokens_metadata_labels.append(
                tokens_labels_and_metadata)

        # namedtuple of label and possible start label if token
        # inside any span, label "O" won't be in  any span
        labels_bit_vector = [LabelSpan("O", "NA") for i in range(
            len(program_tokens_and_metadata))]

        results_and_supporting_facts = []
        # first add results, as they should not be missed
        for result_span in results:
            results_and_supporting_facts.append(Span("RESULT_SPAN",
                                                     result_span.start_line,
                                                     result_span.start_column,
                                                     result_span.end_line,
                                                     result_span.end_column))
        # then add supporting facts
        for result_span in results:
            for supporting_fact_location in result_span.supporting_fact_locations:
                # supporting fact for one result can overlap with result for
                # another result or already added supporting fact span
                # following check avoids such scenario
                possible_overlap = False
                supporting_fact_span = Span("SUPPORTING_FACT_SPAN",
                                            supporting_fact_location.start_line,
                                            supporting_fact_location.start_column,
                                            supporting_fact_location.end_line,
                                            supporting_fact_location.end_column)
                for added_span in results_and_supporting_facts:
                    if (supporting_fact_overlaps(supporting_fact_span, added_span)
                            or supporting_fact_overlaps(added_span, supporting_fact_span)):
                        possible_overlap = True
                        break
                if not possible_overlap:
                    results_and_supporting_facts.append(supporting_fact_span)

        if(positive_or_negative_examples == "positive"):
            for j in range(len(results_and_supporting_facts)):
                # Cubert tokenizer starts counting from 0 and
                # CodeQL results start counting from 1.
                span_type = results_and_supporting_facts[j].span_type
                start_line = results_and_supporting_facts[j].start_line - 1
                end_line = results_and_supporting_facts[j].end_line - 1
                start_column = results_and_supporting_facts[j].start_col - 1
                end_column = results_and_supporting_facts[j].end_col

                is_result_span_missed = True
                for k in range(len(labels_bit_vector)):
                    # span_start needs to be changed wherever label changed to "I"
                    if(start_line == end_line):
                        if(program_tokens_and_metadata[k].metadata.start.
                           line == start_line and program_tokens_and_metadata[k].
                           metadata.end.line == start_line):
                            if(start_column <= program_tokens_and_metadata[k].
                               metadata.start.column and end_column >= program_tokens_and_metadata[k].
                               metadata.end.column):
                                labels_bit_vector[k].label = "I"
                                if span_type == "RESULT_SPAN":
                                    labels_bit_vector[k].span_start = "B"
                                elif span_type == "SUPPORTING_FACT_SPAN":
                                    labels_bit_vector[k].span_start = "F"
                                is_result_span_missed = False
                    else:
                        if(program_tokens_and_metadata[k].metadata.start.line == start_line):
                            if(start_column <= program_tokens_and_metadata[k].
                               metadata.start.column):
                                labels_bit_vector[k].label = "I"
                                if span_type == "RESULT_SPAN":
                                    labels_bit_vector[k].span_start = "B"
                                elif span_type == "SUPPORTING_FACT_SPAN":
                                    labels_bit_vector[k].span_start = "F"
                                is_result_span_missed = False

                        elif(program_tokens_and_metadata[k].metadata.end.line == end_line):
                            if(end_column >= program_tokens_and_metadata[k].metadata.
                               end.column):
                                labels_bit_vector[k].label = "I"
                                if span_type == "RESULT_SPAN":
                                    labels_bit_vector[k].span_start = "B"
                                elif span_type == "SUPPORTING_FACT_SPAN":
                                    labels_bit_vector[k].span_start = "F"
                                is_result_span_missed = False

                        elif(start_line < program_tokens_and_metadata[k].metadata.start.
                                line and end_line > program_tokens_and_metadata[k].
                                metadata.start.line):
                            labels_bit_vector[k].label = "I"
                            if span_type == "RESULT_SPAN":
                                labels_bit_vector[k].span_start = "B"
                            elif span_type == "SUPPORTING_FACT_SPAN":
                                labels_bit_vector[k].span_start = "F"
                            is_result_span_missed = False

                if(is_result_span_missed):
                    missed_spans = missed_spans + 1

        if(positive_or_negative_examples == "positive"):
            if(labels_bit_vector[0].label == "I"):
                if (labels_bit_vector[0].span_start == "B"):
                    labels_bit_vector[0].label = "B"
                elif (labels_bit_vector[0].span_start == "F"):
                    labels_bit_vector[0].label = "F"

            for j in range(1, len(labels_bit_vector)):
                if(labels_bit_vector[j - 1].label == "O"
                        and labels_bit_vector[j].label == "I"):
                    if (labels_bit_vector[j].span_start == "B"):
                        labels_bit_vector[j].label = "B"
                    elif (labels_bit_vector[j].span_start == "F"):
                        labels_bit_vector[j].label = "F"

        span_labels_bit_vector = [labelspan.label for labelspan in labels_bit_vector]
        file_path = raw_merged_query_result_dataset.query_and_files_results[i].raw_file_path.file_path
        err_msg = (str(len(results)) + " " + str(span_labels_bit_vector.count("B")) + " " + query_name
                   + "\n" + str(file_path)
                   + "\n" + str(results))
        assert len(tokenized_program_query_labels.tokens_metadata_labels) == len(span_labels_bit_vector)

        try:
            if(positive_or_negative_examples == "positive"):
                assert "B" in span_labels_bit_vector, err_msg
            elif(positive_or_negative_examples == "negative"):
                assert "B" not in span_labels_bit_vector, err_msg
                assert "I" not in span_labels_bit_vector, err_msg
            # Following assertion will not hold, when
            # there's an overlapping of spans

            # assert len(results) == span_labels_bit_vector.count("B"), err_msg

            # # engulfed_spans doesn't make sense with supporting fact spans
            # engulfed_spans = engulfed_spans + (len(results) - span_labels_bit_vector.count("B"))

            for j in range(len(tokenized_program_query_labels.tokens_metadata_labels)):
                tokenized_program_query_labels.tokens_metadata_labels[j].label = (
                    dataset_with_context_pb2.OutputLabels.Value(span_labels_bit_vector[j]))

            tokenized_program_query_labels_dataset.tokens_and_labels.append(
                tokenized_program_query_labels)
        except AssertionError:
            print(err_msg)

    # print('Overlapped spans: ', engulfed_spans)
    print('Missed/possibly wrong spans', missed_spans)
    return tokenized_program_query_labels_dataset
