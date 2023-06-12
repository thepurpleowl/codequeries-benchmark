import dataset_with_context_pb2
from contexts.get_context import get_all_context, ContextRetrievalError
from contexts.basecontexts import STUB
from create_tokenized_files_labels import get_tokenized_stub_content, get_tokenized_class_header
import hashlib
from tqdm import tqdm

block_ambiguity_dict = dict()


def create_unique_block_id(source_code_file_path, block_body,
                           block_start_line, block_end_line, block_relevance, block_type):
    id = hashlib.md5((
        source_code_file_path + block_body
        + str(block_start_line) + str(block_end_line)  # + str(block_relevance)
        + str(block_type)).
        encode("utf-8")).digest()

    return id


def create_blocks_labels_dataset(tokenized_query_program_labels_dataset,
                                 tree_sitter_parser, keep_header, aux_result_df):
    """
    This function extracts blocks from tokenized labelled program files
    and creates blocks with labels datasets.
    Args:
        tokenized_query_program_labels_dataset: TokenizedQueryProgramLabelsDataset
        protobuf
        tree_sitter_parser: Tree sitter parser object
        keep_header: whether to keep block header info, default True
        aux_result_df: auxiliary query results dataframe
    Returns:
          TokenizedBlockQueryLabelsDataset protobuf
    """
    block_query_labels_dataset = (dataset_with_context_pb2.
                                  TokenizedBlockQueryLabelsDataset())
    example_type = tokenized_query_program_labels_dataset.example_type

    for i in tqdm(range(len(tokenized_query_program_labels_dataset.
                            tokens_and_labels)), desc="Blocks_labels"):
        program_content = (tokenized_query_program_labels_dataset.
                           tokens_and_labels[i].query_and_files_results.
                           raw_file_path.file_content)

        tokens_and_metadata = (tokenized_query_program_labels_dataset.
                               tokens_and_labels[i].tokens_metadata_labels)

        source_code_file_path = (tokenized_query_program_labels_dataset.
                                 tokens_and_labels[i].query_and_files_results.
                                 raw_file_path.file_path.
                                 dataset_file_path.unique_file_path)

        for j in range(len(tokenized_query_program_labels_dataset.
                           tokens_and_labels[i].query_and_files_results.
                           resultlocation)):
            result_location = (tokenized_query_program_labels_dataset.
                               tokens_and_labels[i].query_and_files_results.
                               resultlocation[j])
            message = (tokenized_query_program_labels_dataset.
                       tokens_and_labels[i].query_and_files_results.
                       resultlocation[j].message)
            query_name = (tokenized_query_program_labels_dataset.tokens_and_labels[i].
                          query_and_files_results.query.metadata.name)

            try:
                blocks_and_metadata_list = get_all_context(query_name, program_content, tree_sitter_parser,
                                                           source_code_file_path, message, result_location,
                                                           aux_result_df)
            except ContextRetrievalError as error:
                error_att = error.args[0]
                print('msg: ', error_att['message'], '\n',
                      'type: ', error_att['type'])

                err_dict = {'Error': error_att, 'name': query_name,
                            'file_path': source_code_file_path, 'message': message,
                            'start_line': result_location.start_line, 'end_line': result_location.end_line}
                with open('context_retrieval.log', 'a') as logger:
                    logger.write(str(err_dict) + '\n')

                # Ignore where there is exception
                # This will happen if there is a deviation from how normally
                # CodeQL query flags span - 1 instance in eth_py_open dataset
                continue

            group_of_relevant_and_context_blocks = (dataset_with_context_pb2.
                                                    TokenizedBlockQueryLabelsGroup())

            for a in range(len(blocks_and_metadata_list)):
                block_proto = dataset_with_context_pb2.ProgramBlockDetails()

                block_query_labels_group_item = (dataset_with_context_pb2.
                                                 TokenizedBlockQueryLabels())

                block = blocks_and_metadata_list[a].content

                start_block = blocks_and_metadata_list[a].start_line
                end_block = blocks_and_metadata_list[a].end_line
                otherlines = blocks_and_metadata_list[a].other_lines
                block_relevance = blocks_and_metadata_list[a].relevant
                block_type = (dataset_with_context_pb2.ProgramBlockTypes.
                              Value(blocks_and_metadata_list[a].block_type))

                # for all Block types(STUB also),
                # add class_header tokens at beginning
                if(block_type == dataset_with_context_pb2.ProgramBlockTypes.Value(STUB)):
                    if(keep_header):
                        class_header_token_metadata = get_tokenized_class_header(
                            blocks_and_metadata_list[a].block_header)
                        for k in range(len(class_header_token_metadata)):
                            (block_query_labels_group_item.
                                block_metadata_tokens_labels.append(
                                    class_header_token_metadata[k]))

                    stub_token_metadata = get_tokenized_stub_content(
                        blocks_and_metadata_list[a])
                    for k in range(len(stub_token_metadata)):
                        (block_query_labels_group_item.
                            block_metadata_tokens_labels.append(
                                stub_token_metadata[k]))
                else:
                    if(keep_header):
                        class_header_token_metadata = get_tokenized_class_header(
                            blocks_and_metadata_list[a].block_header)
                        for k in range(len(class_header_token_metadata)):
                            (block_query_labels_group_item.
                                block_metadata_tokens_labels.append(
                                    class_header_token_metadata[k]))

                    for k in range(len(tokens_and_metadata)):
                        start_token = tokens_and_metadata[k].start_line
                        end_token = tokens_and_metadata[k].end_line

                        local_block_lines = otherlines
                        if(not local_block_lines):
                            local_block_lines = [i for i in range(start_block, end_block + 1)]

                        # relax condition with `or`
                        if((start_token in local_block_lines) or (end_token in local_block_lines)):
                            (block_query_labels_group_item.
                                block_metadata_tokens_labels.append(
                                    tokens_and_metadata[k]))

                block_id = create_unique_block_id(
                    source_code_file_path, block,
                    start_block, end_block, block_relevance, block_type)

                block_proto.unique_block_id = block_id
                block_proto.start_line = start_block
                block_proto.end_line = end_block
                block_proto.other_lines.extend(otherlines)
                block_proto.content = block
                block_proto.file_path.CopyFrom(tokenized_query_program_labels_dataset.
                                               tokens_and_labels[i].
                                               query_and_files_results.
                                               raw_file_path.file_path)
                block_proto.metadata = blocks_and_metadata_list[a].metadata
                block_proto.block_type = block_type

                if(blocks_and_metadata_list[a].relevant):
                    block_proto.relevance_label = (dataset_with_context_pb2.BlockRelevance.
                                                   Value("yes"))
                else:
                    block_proto.relevance_label = (dataset_with_context_pb2.BlockRelevance.
                                                   Value("no"))

                block_query_labels_group_item.block.CopyFrom(block_proto)

                block_query_labels_group_item.query_id = (
                    tokenized_query_program_labels_dataset.
                    tokens_and_labels[i].query_and_files_results.query.queryID
                )

                block_query_labels_group_item.distributable = (
                    tokenized_query_program_labels_dataset.
                    tokens_and_labels[i].query_and_files_results.query.distributable
                )

                block_query_labels_group_item.query_name_tokens.extend(
                    tokenized_query_program_labels_dataset.tokens_and_labels[i].
                    query_name_tokens
                )

                block_query_labels_group_item.raw_file.CopyFrom(
                    tokenized_query_program_labels_dataset.
                    tokens_and_labels[i].query_and_files_results.raw_file_path)

                group_of_relevant_and_context_blocks.tokenized_block_query_labels_group_item.append(
                    block_query_labels_group_item
                )

            if(group_of_relevant_and_context_blocks.tokenized_block_query_labels_group_item[0].
               distributable == 1):
                assert len(group_of_relevant_and_context_blocks.
                           tokenized_block_query_labels_group_item) >= 1
                # assert (group_of_relevant_and_context_blocks.
                #         tokenized_block_query_labels_group_item[0].block.
                #         relevance_label) == (dataset_with_context_pb2.BlockRelevance.
                #                              Value("yes"))

            # check for block ambiguity and add
            is_ambiguous, block_group_id = check_block_level_ambiguity(
                group_of_relevant_and_context_blocks)
            if(is_ambiguous):
                block_groups = (block_query_labels_dataset.
                                tokenized_block_query_labels_item)
                for block_group in block_groups:
                    query_id = (
                        block_group.tokenized_block_query_labels_group_item[0].query_id).hex()
                    file_path = (block_group.tokenized_block_query_labels_group_item[0].
                                 raw_file.file_path.dataset_file_path.unique_file_path)
                    block_ids = ''
                    block_id_list = list()
                    for block_item in (block_group.tokenized_block_query_labels_group_item):
                        block_id_list.append(
                            (block_item.block.unique_block_id).hex())
                    for block_id in sorted(block_id_list):
                        block_ids = block_ids + block_id

                    id = hashlib.md5(
                        (query_id + file_path + block_ids).encode("utf-8")).digest()

                    if(id == block_group_id):
                        # previously existing same group of Blocks - G_a
                        ref_blocks = (block_group.
                                      tokenized_block_query_labels_group_item)
                        # current group of Blocks - G_b => here (G_a == G_b)
                        blocks = (group_of_relevant_and_context_blocks.
                                  tokenized_block_query_labels_group_item)
                        # update labels in previously added blocks
                        for block in blocks:
                            for ref_block in ref_blocks:
                                if(block.block.unique_block_id == ref_block.block.unique_block_id):
                                    for tok_block, ref_tok_block in zip(block.block_metadata_tokens_labels,
                                                                        ref_block.block_metadata_tokens_labels):
                                        if(ref_tok_block.label == (dataset_with_context_pb2.
                                                                   OutputLabels.Value("O"))
                                                and tok_block.label != (dataset_with_context_pb2.
                                                                        OutputLabels.Value("O"))):
                                            ref_tok_block.label = tok_block.label
            else:
                block_query_labels_dataset.tokenized_block_query_labels_item.append(
                    group_of_relevant_and_context_blocks)
                block_query_labels_dataset.example_types.append(example_type)

    return block_query_labels_dataset


def check_block_level_ambiguity(group_of_relevant_and_context_blocks):
    """
    This function check block level ambiguity for a given TokenizedBlockQueryLabelsGroup,
    considering prepared TokenizedBlockQueryLabelsDataset till that point.
    Args:
        group_of_relevant_and_context_blocks: TokenizedBlockQueryLabelsGroup
        protobuf
    Returns:
        ambiguity_status, unique TokenizedBlockQueryLabelsGroup id
    """
    query_id = (group_of_relevant_and_context_blocks.
                tokenized_block_query_labels_group_item[0].
                query_id).hex()
    file_path = (group_of_relevant_and_context_blocks.
                 tokenized_block_query_labels_group_item[0].raw_file.
                 file_path.dataset_file_path.unique_file_path)
    block_ids = ''
    block_id_list = list()
    for block_item in (group_of_relevant_and_context_blocks.
                       tokenized_block_query_labels_group_item):
        block_id_list.append((block_item.block.unique_block_id).hex())
    for block_id in sorted(block_id_list):
        block_ids = block_ids + block_id

    id = hashlib.md5(
        (query_id + file_path + block_ids).encode("utf-8")).digest()
    block_id_set = set(block_id_list)

    if id in block_ambiguity_dict:
        return True, id
    else:
        block_ambiguity_dict[id] = block_id_set
        return False, id
