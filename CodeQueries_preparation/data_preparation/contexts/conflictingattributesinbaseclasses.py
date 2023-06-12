from basecontexts import BaseContexts, CLASS_FUNCTION
import re


class ConflictingAttributesInBaseClasses(BaseContexts):
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

    context_object = ConflictingAttributesInBaseClasses(parser)

    local_block = context_object.get_local_block(program_content, start_line, end_line)
    # local_block will be always a Block outside parent_class_blocks
    local_block.relevant = True
    required_blocks = [local_block]

    _, parent_class_blocks = context_object.get_local_and_super_class(program_content, start_line, end_line)
    all_blocks = context_object.get_all_blocks(program_content)
    for block in all_blocks:
        for p_block in parent_class_blocks:
            if (block.start_line >= p_block.start_line
                    and block.end_line <= p_block.end_line):
                required_blocks.append(block)

    # MRO only for conflicting attribute
    conflicting_attributes = []
    for msg in message.split('\n'):
        attr = re.findall(r"\'(.*)\':", msg)
        if attr:
            conflicting_attributes.append(attr[0].strip())

    for class_block in parent_class_blocks:
        cls = class_block.metadata.split('.')[-1]
        for conflicting_attr in conflicting_attributes:
            mro_func_block = context_object.get_mro_function_block(conflicting_attr,
                                                                   cls, program_content)
            if(mro_func_block is not None):
                if(mro_func_block in required_blocks):
                    for block in required_blocks:
                        if(block == mro_func_block):
                            block.relevant = True
                else:
                    mro_func_block.relevant = True
                    required_blocks.append(mro_func_block)

    # mark relevance
    lines = program_content.split('\n')
    for block in required_blocks:
        # function level check
        if(block.block_type == CLASS_FUNCTION):
            # check if function name is __init__
            func_name = block.metadata.split('.')[-1]
            if(func_name == '__init__'):
                block.relevant = True

            # check for equal functions in all classes
            for o_block in required_blocks:
                if(o_block.block_type == CLASS_FUNCTION and block != o_block):
                    o_func_name = o_block.metadata.split('.')[-1]
                    if(func_name == o_func_name
                            and func_name not in ['__init__', 'process_request']):
                        block.relevant = True
                        o_block.relevant = True
        # class field level checks
        else:
            # as these are class block only possible bock_types are CLASS_FUNCTION & CLASS_OTHER
            # hence check for other lines covers whole class body
            for i in block.other_lines:
                for conflicting_attr in conflicting_attributes:
                    if(('self.' + conflicting_attr) in lines[i]
                            or conflicting_attr in lines[i]):
                        block.relevant = True
                        break
                if(block.relevant):
                    break

    return required_blocks
