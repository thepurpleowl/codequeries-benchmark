from basecontexts import BaseContexts, CLASS_FUNCTION


class DefineEqualsWhenAddingAttributes(BaseContexts):
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

    context_object = DefineEqualsWhenAddingAttributes(parser)
    local_class_block, parent_class_blocks = context_object.get_local_and_super_class(program_content, start_line, end_line)
    all_blocks = context_object.get_all_blocks(program_content)

    required_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)
    # sanity check if
    if local_class_block is not None:
        for block in all_blocks:
            # if a block in base class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                if(block == local_block):
                    block.relevant = True
                required_blocks.append(block)
            # if a block in any of the parent class
            else:
                for p_block in parent_class_blocks:
                    if(block.start_line >= p_block.start_line
                            and block.end_line <= p_block.end_line):
                        required_blocks.append(block)

        local_class = local_class_block.metadata.split('.')[-1]
        mro_eq_block = context_object.get_mro_function_block('__eq__',
                                                             local_class,
                                                             program_content)

        # for __eq__ : with MRO
        # by definition of query, there has to be an mro_eq_block, i.e., __eq__()
        # from some super class, but not the case bcs of single file restriction
        if(mro_eq_block is not None):
            if(mro_eq_block in required_blocks):
                for block in required_blocks:
                    if(block == mro_eq_block):
                        block.relevant = True
            else:
                mro_eq_block.relevant = True
                required_blocks.append(mro_eq_block)

        # for __init__ : __init__ specific resolution - all __init__ in required_blocks are relevant
        # super_class = None
        # super_class = context_object.check_multiple_inheritance_super(program_content,
        #                                                               local_class_block)

        # if(super_class is None):
        #     super_class = local_class_block.metadata.split('.')[-1]

        for block in required_blocks:
            if(block.block_type == CLASS_FUNCTION):
                func_name = block.metadata.split('.')[-1]

                if(func_name == '__init__'):
                    block.relevant = True

    return required_blocks
