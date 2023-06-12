from basecontexts import BaseContexts, CLASS_FUNCTION


class MissingCallToInit(BaseContexts):
    """
    This module extracts module level context and corrsponding metadata from
    program content.
    Metadata contains start/end line number informations for corresponding context.
    """
    def __init__(self, parser):
        """
        Args:
            parser: Tree sitter parser object
        """
        super().__init__(parser)


def get_query_specific_context(program_content, parser, file_path, message, result_span, aux_result_df=None):
    """
    This functions returns relevant Blocks as query specific context.
    Args:
        program_content: Program in string format from which we need to
        extract classes
        parser : tree_sitter_parser
        file_path : file of program_content
        message : CodeQL message
        result_span: CodeQL-treesitter adjusted namedtuple of
                     (start_line, start_col, end_line, end_col)
        aux_result_df:  auxiliary query results dataframe
    Returns:
        A list consisting relevant Blocks
    """
    start_line = result_span.start_line
    end_line = result_span.end_line

    context_object = MissingCallToInit(parser)
    local_class_block, parent_class_blocks = context_object.get_local_and_super_class(program_content, start_line, end_line)
    all_blocks = context_object.get_all_blocks(program_content)
    local_block = context_object.get_local_block(program_content, start_line, end_line)

    required_blocks = []
    # sanity check if
    if local_class_block is not None:
        for block in all_blocks:
            # if a block in base class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                # local block is CLASS_OTHER
                if(block == local_block):
                    block.relevant = True
                    required_blocks.append(block)
                elif(block.block_type == CLASS_FUNCTION
                        and block.metadata.split('.')[-1] == '__init__'):
                    block.relevant = True
                    required_blocks.append(block)
            # if a block in any of the parent class
            else:
                for p_block in parent_class_blocks:
                    if(block.start_line >= p_block.start_line
                            and block.end_line <= p_block.end_line):
                        if(block.block_type == CLASS_FUNCTION
                                and block.metadata.split('.')[-1] == '__init__'):
                            block.relevant = True
                            required_blocks.append(block)

    return required_blocks
