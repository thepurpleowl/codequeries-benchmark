import numpy as np
from tqdm import tqdm
import dataset_with_context_pb2
from tensor2tensor.data_generators import text_encoder
from transformers import RobertaTokenizer


class SubTokenize:
    def __init__(self, vocab_file):
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(vocab_file)

    def get_subtokens(self, token) -> list:
        encoded_tokens = self.subword_tokenizer.encode_without_tokenizing(
            token)
        decoded_tokens = self.subword_tokenizer.decode_list(encoded_tokens)
        return decoded_tokens


def create_cubert_subtokens_labels(ordering_of_blocks,
                                   tokenized_block_query_labels_dataset,
                                   vocab_file):
    """
    This function creates subtokens of program tokens and query tokens for CuBERT model.
    Args:
        ordering_of_blocks: specific ordering among blocks
        tokenized_block_query_labels_dataset: TokenizedBlockQueryLabelsDataset protobuf
        vocab_file: model vocab file
    Returns:
        BlockQuerySubtokensLabelsDataset protobuf
    """
    subtokenize_object = SubTokenize(vocab_file)

    block_query_subtokens_labels_dataset = (
        dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset()
    )

    for i in tqdm(range(len(tokenized_block_query_labels_dataset.
                            tokenized_block_query_labels_item)), desc="Block_subtokens_labels"):

        block_query_subtokens_labels_group = (
            dataset_with_context_pb2.
            BlockQuerySubtokensLabelsGroup()
        )

        example_type = (tokenized_block_query_labels_dataset.example_types[i])

        for j in range(len(tokenized_block_query_labels_dataset.
                       tokenized_block_query_labels_item[i].
                       tokenized_block_query_labels_group_item)):

            block_query_subtokens_labels_group_item = (
                dataset_with_context_pb2.
                BlockQuerySubtokensLabels()
            )

            block_query_subtokens_labels_group_item.query_id = (
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].query_id
            )

            block_query_subtokens_labels_group_item.distributable = (
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].distributable
            )

            block_query_subtokens_labels_group_item.raw_file.CopyFrom(
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].raw_file
            )

            block_query_subtokens_labels_group_item.block.CopyFrom(
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].block
            )

            query_subtokens = []

            for k in range(len(tokenized_block_query_labels_dataset.
                               tokenized_block_query_labels_item[i].
                               tokenized_block_query_labels_group_item[j].
                               query_name_tokens)):
                token = (tokenized_block_query_labels_dataset.
                         tokenized_block_query_labels_item[i].
                         tokenized_block_query_labels_group_item[j].
                         query_name_tokens[k])
                subtokens = subtokenize_object.get_subtokens(token)
                query_subtokens.extend(subtokens)

            block_query_subtokens_labels_group_item.query_name_subtokens.extend(
                query_subtokens)

            for k in range(len(tokenized_block_query_labels_dataset.
                               tokenized_block_query_labels_item[i].
                               tokenized_block_query_labels_group_item[j].
                               block_metadata_tokens_labels)):
                token_and_metadata = (tokenized_block_query_labels_dataset.
                                      tokenized_block_query_labels_item[i].
                                      tokenized_block_query_labels_group_item[j].
                                      block_metadata_tokens_labels[k])
                program_token = token_and_metadata.program_token
                label = token_and_metadata.label
                subtokens = subtokenize_object.get_subtokens(program_token)
                for m in range(len(subtokens)):
                    subtoken_label_item = dataset_with_context_pb2.SubtokenLabel()
                    subtoken_label_item.program_subtoken = subtokens[m]
                    subtoken_label_item.label = label
                    block_query_subtokens_labels_group_item.block_subtokens_labels.append(
                        subtoken_label_item)
                    if(label == dataset_with_context_pb2.OutputLabels.Value("B")
                            or label == dataset_with_context_pb2.OutputLabels.Value("F")):
                        label = dataset_with_context_pb2.OutputLabels.Value(
                            "I")
            block_query_subtokens_labels_group.block_query_subtokens_labels_group_item.append(
                block_query_subtokens_labels_group_item)

        block_query_subtokens_labels_dataset.block_query_subtokens_labels_item.append(
            block_query_subtokens_labels_group
        )
        block_query_subtokens_labels_dataset.example_types.append(
            example_type
        )

    if(ordering_of_blocks == "default_proto"):
        return sort_blocks_proto_order(
            block_query_subtokens_labels_dataset
        )
    elif(ordering_of_blocks == "line_number"):
        return sort_blocks_line_number_order(
            block_query_subtokens_labels_dataset
        )

    else:
        return block_query_subtokens_labels_dataset


def get_codebert_filter_status(token):
    """
    This function checks if token should be filtered CodeBERT model.
    Args:
        token: TokenizedBlockQueryLabelsDataset protobuf
    Returns:
        True/False - should be filtered out
    """
    filter_out = False
    __TOKEN_TYPES_TO_BE_FILTERED__ = ['___NL___', '___NEWLINE___', '___DEDENT___', '___INDENT___']
    for token_type in __TOKEN_TYPES_TO_BE_FILTERED__:
        if(token.startswith(token_type)):
            filter_out = True
            break
    return filter_out


def create_codebert_subtokens_labels(ordering_of_blocks,
                                     tokenized_block_query_labels_dataset):
    """
    This function creates subtokens of program tokens and query tokens for CodeBERT model.
    Args:
        ordering_of_blocks: specific ordering among blocks
        tokenized_block_query_labels_dataset: TokenizedBlockQueryLabelsDataset protobuf
    Returns:
        BlockQuerySubtokensLabelsDataset protobuf
    """
    codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    __PREFIX_SPACE__ = ' '

    block_query_subtokens_labels_dataset = (
        dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset()
    )

    for i in tqdm(range(len(tokenized_block_query_labels_dataset.
                            tokenized_block_query_labels_item)), desc="Block_subtokens_labels"):

        block_query_subtokens_labels_group = (
            dataset_with_context_pb2.
            BlockQuerySubtokensLabelsGroup()
        )

        example_type = (tokenized_block_query_labels_dataset.example_types[i])

        for j in range(len(tokenized_block_query_labels_dataset.
                       tokenized_block_query_labels_item[i].
                       tokenized_block_query_labels_group_item)):

            block_query_subtokens_labels_group_item = (
                dataset_with_context_pb2.
                BlockQuerySubtokensLabels()
            )

            block_query_subtokens_labels_group_item.query_id = (
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].query_id
            )

            block_query_subtokens_labels_group_item.distributable = (
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].distributable
            )

            block_query_subtokens_labels_group_item.raw_file.CopyFrom(
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].raw_file
            )

            block_query_subtokens_labels_group_item.block.CopyFrom(
                tokenized_block_query_labels_dataset.
                tokenized_block_query_labels_item[i].
                tokenized_block_query_labels_group_item[j].block
            )

            query_subtokens = []
            is_first_query_subtoken = True
            for k in range(len(tokenized_block_query_labels_dataset.
                               tokenized_block_query_labels_item[i].
                               tokenized_block_query_labels_group_item[j].
                               query_name_tokens)):
                token = (tokenized_block_query_labels_dataset.
                         tokenized_block_query_labels_item[i].
                         tokenized_block_query_labels_group_item[j].
                         query_name_tokens[k])
                # if token should be filtered out,
                # elif whether the first token to be subtokenized
                # else add space
                if(get_codebert_filter_status(token)):
                    continue
                elif(is_first_query_subtoken):
                    is_first_query_subtoken = False
                else:
                    token = __PREFIX_SPACE__ + token
                subtokens = codebert_tokenizer.tokenize(token)
                query_subtokens.extend(subtokens)

            block_query_subtokens_labels_group_item.query_name_subtokens.extend(
                query_subtokens)

            is_first_program_subtoken = True
            prev_token_discarded = False
            prev_discarded_label = None
            for k in range(len(tokenized_block_query_labels_dataset.
                               tokenized_block_query_labels_item[i].
                               tokenized_block_query_labels_group_item[j].
                               block_metadata_tokens_labels)):
                token_and_metadata = (tokenized_block_query_labels_dataset.
                                      tokenized_block_query_labels_item[i].
                                      tokenized_block_query_labels_group_item[j].
                                      block_metadata_tokens_labels[k])
                program_token = token_and_metadata.program_token
                # if token should be filtered out,
                # elif whether the first token to be subtokenized
                # else add space
                if(get_codebert_filter_status(program_token)):
                    # discard status for next token
                    temp_discarded_label = prev_discarded_label
                    prev_discarded_label = token_and_metadata.label
                    # if some "I" after discarded "B" are also discarded
                    # "B" should be preserved as prev_discarded_label
                    if(prev_token_discarded
                            and token_and_metadata.label == (dataset_with_context_pb2.
                                                             OutputLabels.Value("I"))):
                        if(temp_discarded_label == (dataset_with_context_pb2.
                                                    OutputLabels.Value("B"))):
                            prev_discarded_label = (dataset_with_context_pb2.
                                                    OutputLabels.Value("B"))
                        elif(temp_discarded_label == (dataset_with_context_pb2.
                                                      OutputLabels.Value("F"))):
                            prev_discarded_label = (dataset_with_context_pb2.
                                                    OutputLabels.Value("F"))
                    prev_token_discarded = True
                    continue
                elif(is_first_program_subtoken):
                    is_first_program_subtoken = False
                else:
                    program_token = __PREFIX_SPACE__ + program_token
                subtokens = codebert_tokenizer.tokenize(program_token)
                label = token_and_metadata.label

                # check if just prev token discarded had label "B"/"F"
                if(prev_token_discarded
                        and label == (dataset_with_context_pb2.
                                      OutputLabels.Value("I"))):
                    if(prev_discarded_label == (dataset_with_context_pb2.
                                                OutputLabels.Value("B"))):
                        label = dataset_with_context_pb2.OutputLabels.Value("B")
                    elif(prev_discarded_label == (dataset_with_context_pb2.
                                                  OutputLabels.Value("F"))):
                        label = dataset_with_context_pb2.OutputLabels.Value("F")

                # if control flow comes till here, that means the token is not
                # discarded and discard status should be updated for next token
                if(prev_token_discarded):
                    prev_token_discarded = False

                for m in range(len(subtokens)):
                    subtoken_label_item = dataset_with_context_pb2.SubtokenLabel()
                    subtoken_label_item.program_subtoken = subtokens[m]
                    subtoken_label_item.label = label
                    block_query_subtokens_labels_group_item.block_subtokens_labels.append(
                        subtoken_label_item)
                    if(label == dataset_with_context_pb2.OutputLabels.Value("B")
                            or label == dataset_with_context_pb2.OutputLabels.Value("F")):
                        label = dataset_with_context_pb2.OutputLabels.Value(
                            "I")
            block_query_subtokens_labels_group.block_query_subtokens_labels_group_item.append(
                block_query_subtokens_labels_group_item)

        block_query_subtokens_labels_dataset.block_query_subtokens_labels_item.append(
            block_query_subtokens_labels_group
        )
        block_query_subtokens_labels_dataset.example_types.append(
            example_type
        )

    if(ordering_of_blocks == "default_proto"):
        return sort_blocks_proto_order(
            block_query_subtokens_labels_dataset
        )
    elif(ordering_of_blocks == "line_number"):
        return sort_blocks_line_number_order(
            block_query_subtokens_labels_dataset
        )

    else:
        return block_query_subtokens_labels_dataset


def sort_blocks_proto_order(block_query_subtokens_labels_dataset):
    # sorting blocks by default order defined in proto file

    block_query_subtokens_labels_dataset_ordered = (
        dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset()
    )

    for i in tqdm(range(len(block_query_subtokens_labels_dataset.
                            block_query_subtokens_labels_item)), desc="Ordering_blocks"):
        block_types = []

        block_query_subtokens_labels_group = (
            dataset_with_context_pb2.
            BlockQuerySubtokensLabelsGroup()
        )

        for j in range(len(block_query_subtokens_labels_dataset.
                       block_query_subtokens_labels_item[i].
                       block_query_subtokens_labels_group_item)):

            block_types.append(block_query_subtokens_labels_dataset.
                               block_query_subtokens_labels_item[i].
                               block_query_subtokens_labels_group_item[j].
                               block.block_type)

        sorted_indices = np.argsort(block_types)

        for j in range(len(sorted_indices)):
            block_query_subtokens_labels_group.block_query_subtokens_labels_group_item.append(
                block_query_subtokens_labels_dataset.
                block_query_subtokens_labels_item[i].
                block_query_subtokens_labels_group_item[sorted_indices[j]]
            )

        block_query_subtokens_labels_dataset_ordered.block_query_subtokens_labels_item.append(
            block_query_subtokens_labels_group
        )
        block_query_subtokens_labels_dataset_ordered.example_types.append(
            block_query_subtokens_labels_dataset.example_types[i]
        )

    return block_query_subtokens_labels_dataset_ordered


def sort_blocks_line_number_order(block_query_subtokens_labels_dataset):
    block_query_subtokens_labels_dataset_ordered = (
        dataset_with_context_pb2.BlockQuerySubtokensLabelsDataset()
    )

    for i in tqdm(range(len(block_query_subtokens_labels_dataset.
                            block_query_subtokens_labels_item)), desc="Ordering_blocks"):
        block_start_line = []

        block_query_subtokens_labels_group = (
            dataset_with_context_pb2.
            BlockQuerySubtokensLabelsGroup()
        )

        for j in range(len(block_query_subtokens_labels_dataset.
                       block_query_subtokens_labels_item[i].
                       block_query_subtokens_labels_group_item)):

            block_start_line.append(block_query_subtokens_labels_dataset.
                                    block_query_subtokens_labels_item[i].
                                    block_query_subtokens_labels_group_item[j].
                                    block.start_line)

        sorted_indices = np.argsort(block_start_line)

        for j in range(len(sorted_indices)):
            block_query_subtokens_labels_group.block_query_subtokens_labels_group_item.append(
                block_query_subtokens_labels_dataset.
                block_query_subtokens_labels_item[i].
                block_query_subtokens_labels_group_item[sorted_indices[j]]
            )

        block_query_subtokens_labels_dataset_ordered.block_query_subtokens_labels_item.append(
            block_query_subtokens_labels_group
        )
        block_query_subtokens_labels_dataset_ordered.example_types.append(
            block_query_subtokens_labels_dataset.example_types[i]
        )

    return block_query_subtokens_labels_dataset_ordered
