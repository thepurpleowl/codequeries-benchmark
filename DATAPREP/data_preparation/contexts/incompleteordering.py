from basecontexts import BaseContexts, CLASS_FUNCTION


class IncompleteOrdering(BaseContexts):
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

    context_object = IncompleteOrdering(parser)
    local_class_block = context_object.get_local_class(program_content, start_line, end_line)

    all_blocks = context_object.get_all_blocks(program_content)
    required_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)
    if local_class_block is not None:
        order_func_names = set(['__lt__', '__gt__', '__le__', '__ge__'])
        local_order_func_names = set()
        for block in all_blocks:
            # if a block in base class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                if(block == local_block):
                    block.relevant = True
                if(block.block_type == CLASS_FUNCTION):
                    func_name = block.metadata.split('.')[-1]
                    if(func_name in order_func_names):
                        local_order_func_names.add(func_name)
                required_blocks.append(block)

        # add ordering functions from super class if
        # ordering fucntion not in local class
        current_class = local_class_block.metadata.split('.')[-1]
        remaining_order_func_names = order_func_names - local_order_func_names
        if(remaining_order_func_names):
            for func in sorted(remaining_order_func_names):
                mro_func_block = context_object.get_mro_function_block(func,
                                                                       current_class,
                                                                       program_content)
                if(mro_func_block is not None
                        and mro_func_block not in required_blocks):
                    required_blocks.append(mro_func_block)

        # mark relevance
        for block in required_blocks:
            if(block.block_type == CLASS_FUNCTION):
                func_name = block.metadata.split('.')[-1]
                if(func_name in order_func_names):
                    block.relevant = True

    return required_blocks
