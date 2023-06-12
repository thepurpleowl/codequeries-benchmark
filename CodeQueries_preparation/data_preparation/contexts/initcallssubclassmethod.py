from basecontexts import BaseContexts, CLASS_FUNCTION


class InitCallsSubclassMethod(BaseContexts):
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

    context_object = InitCallsSubclassMethod(parser)
    local_class_block, child_class_blocks = context_object.get_local_and_child_class(program_content, start_line, end_line)
    all_blocks = context_object.get_all_blocks(program_content)

    required_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)

    # for intersecting function check
    parent_class_functions = set()
    child_class_functions = dict()  # dictionary of set

    required_parent_class_blocks = []
    required_child_class_blocks = []
    if local_class_block is not None:
        for block in all_blocks:
            # if a block in local class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                if(block == local_block):
                    block.relevant = True
                required_parent_class_blocks.append(block)
                # for intersecting function check
                if(block.block_type == CLASS_FUNCTION):
                    parent_class_functions.add(block.metadata.split('.')[-1])
            # if a block in any of the child class
            else:
                for p_block in child_class_blocks:
                    class_name = p_block.metadata.split('.')[-1]
                    if (block.start_line >= p_block.start_line
                            and block.end_line <= p_block.end_line):
                        required_child_class_blocks.append(block)
                        # for intersecting function check
                        if(block.block_type == CLASS_FUNCTION):
                            func_name = block.metadata.split('.')[-1]
                            if(class_name in child_class_functions):
                                child_class_functions[class_name].add(func_name)
                            else:
                                child_class_functions[class_name] = set([func_name])

        # get intersecting functions
        intersecting_functions = dict()
        if(parent_class_functions):
            for cls, func_set in child_class_functions.items():
                intersecting_functions[cls] = parent_class_functions.intersection(func_set)

        # mark relevance
        init_func = '__init__'
        for block in required_child_class_blocks:
            if(block.block_type == CLASS_FUNCTION):
                class_name = block.metadata.split('.')[-2]
                func_name = block.metadata.split('.')[-1]
                # check if function is not __init__
                # and in intersecting functions
                if(func_name != init_func
                        # for classes without functions/ intersecting functions
                        and class_name in intersecting_functions.keys()
                        and func_name in intersecting_functions[class_name]):
                    block.relevant = True

        for block in required_parent_class_blocks:
            if(block.block_type == CLASS_FUNCTION):
                class_name = block.metadata.split('.')[-2]
                func_name = block.metadata.split('.')[-1]
                # check if function is __init__
                # or in intersecting functions
                if(func_name == init_func):
                    block.relevant = True
                    break

        required_blocks.extend(required_child_class_blocks)
        required_blocks.extend(required_parent_class_blocks)

    return required_blocks
