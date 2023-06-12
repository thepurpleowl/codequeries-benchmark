from basecontexts import BaseContexts, CLASS_FUNCTION


class IterReturnsNonIterator(BaseContexts):
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

    context_object = IterReturnsNonIterator(parser)

    all_blocks = context_object.get_all_blocks(program_content)
    local_class_block = context_object.get_local_class(program_content, start_line, end_line)
    local_block = context_object.get_local_block(program_content, start_line, end_line)

    required_blocks = []
    assert message.split()[0].strip() == 'Class'
    target_class = message.split()[1].strip()
    if local_class_block is not None:
        for block in all_blocks:
            # if a block in local class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                # local block is CLASS_OTHER
                if(block == local_block):
                    block.relevant = True
                    required_blocks.append(block)
                elif(block.block_type == CLASS_FUNCTION
                        and block.metadata.split('.')[-1] == '__iter__'):
                    block.relevant = True
                    required_blocks.append(block)
            # if other class __iter__ returns target_class iterator object
            elif(block.block_type == CLASS_FUNCTION
                    and block.metadata.split('.')[-1] == '__iter__'
                    and ('return ' + target_class) in (' '.join(block.content.split()))):
                block.relevant = True
                required_blocks.append(block)

    return required_blocks
