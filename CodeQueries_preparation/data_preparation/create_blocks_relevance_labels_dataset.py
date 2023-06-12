import dataset_with_context_pb2
from contexts.get_context import get_all_context_with_simplified_relevance, ContextRetrievalError
from contexts.basecontexts import STUB
from create_tokenized_files_labels import get_tokenized_class_header
import hashlib
from tqdm import tqdm

block_ambiguity_dict = dict()


def create_unique_block_id(source_code_file_path, block_body,
                           block_start_line, block_end_line, block_relevance, block_type):
    # still generating block_id without relevance, because
    #   i.  there's no ambiguity check
    #   ii. if cross-reference required with Oracle data
    id = hashlib.md5((
        source_code_file_path + block_body
        + str(block_start_line) + str(block_end_line)  # + str(block_relevance)
        + str(block_type)).
        encode("utf-8")).digest()

    return id


def create_blocks_relevance_labels_dataset(tokenized_query_program_labels_dataset,
                                           tree_sitter_parser, keep_header,
                                           aux_result_df, only_relevant_blocks):
    """
    This function extracts blocks from tokenized labelled program files
    and creates blocks with labels datasets.
    Args:
        tokenized_query_program_labels_dataset: TokenizedQueryProgramLabelsDataset
        protobuf
        tree_sitter_parser: Tree sitter parser object
        keep_header: whether to keep block header info, default True
        aux_result_df: auxiliary query results dataframe
        only_relevant_blocks: whether to use only relevant blocks(for Upper Bound)
                              or all blocks(for relevance prediction model)
    Returns:
          TokenizedBlockQueryLabelsDataset protobuf
    """
    # dictionary to be used for aggregating context blocks, of the form
    # {(file_path, query_name): {block_key: Block, ...}, ...}
    visited_pairs = {}

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
        query_name = (tokenized_query_program_labels_dataset.tokens_and_labels[i].
                      query_and_files_results.query.metadata.name)
        # add visited (file, query)
        file_query_pair_key = (source_code_file_path, query_name)
        visited_pairs[file_query_pair_key] = {}

        for j in range(len(tokenized_query_program_labels_dataset.
                           tokens_and_labels[i].query_and_files_results.
                           resultlocation)):
            result_location = (tokenized_query_program_labels_dataset.
                               tokens_and_labels[i].query_and_files_results.
                               resultlocation[j])
            message = (tokenized_query_program_labels_dataset.
                       tokens_and_labels[i].query_and_files_results.
                       resultlocation[j].message)

            try:
                # both relevant and irrelevant block for a result location
                blocks_and_metadata_list = get_all_context_with_simplified_relevance(query_name, program_content,
                                                                                     tree_sitter_parser,
                                                                                     source_code_file_path,
                                                                                     message, result_location,
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

            # aggregate context blocks and relevance
            stub_blocks = 0
            for block in blocks_and_metadata_list:
                block_key = (block.metadata + '_' + block.block_header + '_'
                             + str(block.start_line) + '_'
                             + str(block.end_line))
                if block_key in visited_pairs[file_query_pair_key]:
                    if(not visited_pairs[file_query_pair_key][block_key].relevant
                            and block.relevant):
                        visited_pairs[file_query_pair_key][block_key].relevant = block.relevant
                else:
                    if(block.block_type != STUB):
                        visited_pairs[file_query_pair_key][block_key] = block
                    else:
                        stub_blocks += 1
            # check for consistency in number of total blocks
            try:
                assert (len(blocks_and_metadata_list) - stub_blocks) == len(visited_pairs[file_query_pair_key])
            except AssertionError:
                print(len(blocks_and_metadata_list), len(visited_pairs[file_query_pair_key]))
                with open('aggregation.log', 'a') as logger:
                    logger.write(str(blocks_and_metadata_list) + '\n -------------- \n')
                    logger.write(str(visited_pairs[file_query_pair_key]) + '\n -------------- \n')

        # if context extraction error would have occured
        # for all result locations - ignore the example
        if(len(visited_pairs[file_query_pair_key]) == 0):
            continue

        total_blocks = len(visited_pairs[file_query_pair_key])
        total_relevant_blocks = 0
        for _, block in visited_pairs[file_query_pair_key].items():
            if(block.relevant):
                total_relevant_blocks += 1
        # ---------------------------------------------
        # form block label data with aggregated context
        # ---------------------------------------------
        group_of_relevant_and_context_blocks = (dataset_with_context_pb2.
                                                TokenizedBlockQueryLabelsGroup())

        aggregated_blocks_and_metadata_list = visited_pairs[file_query_pair_key]
        for _, block_key in enumerate(aggregated_blocks_and_metadata_list):
            if(only_relevant_blocks == 'yes'
                    and not aggregated_blocks_and_metadata_list[block_key].relevant):
                continue
            if(only_relevant_blocks == 'yes'):
                assert aggregated_blocks_and_metadata_list[block_key].relevant

            block_proto = dataset_with_context_pb2.ProgramBlockDetails()

            block_query_labels_group_item = (dataset_with_context_pb2.
                                             TokenizedBlockQueryLabels())

            block_content = aggregated_blocks_and_metadata_list[block_key].content

            start_block = aggregated_blocks_and_metadata_list[block_key].start_line
            end_block = aggregated_blocks_and_metadata_list[block_key].end_line
            otherlines = aggregated_blocks_and_metadata_list[block_key].other_lines
            block_relevance = aggregated_blocks_and_metadata_list[block_key].relevant
            block_type = (dataset_with_context_pb2.ProgramBlockTypes.
                          Value(aggregated_blocks_and_metadata_list[block_key].block_type))

            # add class_header tokens at beginning
            if(keep_header):
                class_header_token_metadata = get_tokenized_class_header(
                    aggregated_blocks_and_metadata_list[block_key].block_header)
                for k in range(len(class_header_token_metadata)):
                    (block_query_labels_group_item.
                        block_metadata_tokens_labels.append(
                            class_header_token_metadata[k]))

            for k in range(len(tokens_and_metadata)):
                start_token = tokens_and_metadata[k].start_line
                end_token = tokens_and_metadata[k].end_line

                local_block_lines = otherlines
                if(not local_block_lines):
                    local_block_lines = [li for li in range(start_block, end_block + 1)]

                # relax condition with `or`
                if((start_token in local_block_lines) or (end_token in local_block_lines)):
                    (block_query_labels_group_item.
                        block_metadata_tokens_labels.append(
                            tokens_and_metadata[k]))

            block_id = create_unique_block_id(
                source_code_file_path, block_content,
                start_block, end_block, block_relevance, block_type)

            block_proto.unique_block_id = block_id
            block_proto.start_line = start_block
            block_proto.end_line = end_block
            block_proto.other_lines.extend(otherlines)
            block_proto.content = block_content
            block_proto.file_path.CopyFrom(tokenized_query_program_labels_dataset.
                                           tokens_and_labels[i].
                                           query_and_files_results.
                                           raw_file_path.file_path)
            block_proto.metadata = aggregated_blocks_and_metadata_list[block_key].metadata
            block_proto.block_type = block_type

            if(aggregated_blocks_and_metadata_list[block_key].relevant):
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

        if(only_relevant_blocks == 'yes'):
            assert (len(group_of_relevant_and_context_blocks.tokenized_block_query_labels_group_item)
                    == total_relevant_blocks)
        else:
            assert (len(group_of_relevant_and_context_blocks.tokenized_block_query_labels_group_item)
                    == total_blocks)

        # add to dataset
        block_query_labels_dataset.tokenized_block_query_labels_item.append(
            group_of_relevant_and_context_blocks)
        block_query_labels_dataset.example_types.append(example_type)

    return block_query_labels_dataset
