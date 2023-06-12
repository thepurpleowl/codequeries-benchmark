from basecontexts import BaseContexts, CLASS_FUNCTION


class EqualsOrHash(BaseContexts):
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

    context_object = EqualsOrHash(parser)
    local_class_block = context_object.get_local_class(program_content, start_line, end_line)

    eq_or_hash = set(['__eq__', '__hash__'])
    not_implemented = set([message.split()[-1].split('.')[0].strip()])
    implemented = next(iter(eq_or_hash - not_implemented))
    not_implemented = next(iter(not_implemented))

    all_blocks = context_object.get_all_blocks(program_content)
    required_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)
    if local_class_block is not None:
        for block in all_blocks:
            # if a block in the class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                if(block == local_block):
                    block.relevant = True
                elif(block.block_type == CLASS_FUNCTION
                        and block.metadata.split('.')[-1] == implemented):
                    block.relevant = True
                required_blocks.append(block)

        # add `not_implemented` from super class with MRO
        current_class = local_class_block.metadata.split('.')[-1]
        mro_func_block = context_object.get_mro_function_block(not_implemented,
                                                               current_class,
                                                               program_content)
        if(mro_func_block is not None):
            mro_func_block.relevant = True
            if(mro_func_block not in required_blocks):
                required_blocks.append(mro_func_block)
            # else block shouldn't occur, as the class
            # hasn't implemented the `not_implemented`

    return required_blocks
