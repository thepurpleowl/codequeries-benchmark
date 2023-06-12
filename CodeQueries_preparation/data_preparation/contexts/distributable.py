from basecontexts import BaseContexts


class Distributable(BaseContexts):
    """
    This module extracts Blocks and corrsponding metadata from program content for
    distributable queries.
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
    context_object = Distributable(parser)

    local_block = context_object.get_local_block(program_content, start_line, end_line)
    local_block.relevant = True
    required_blocks = [local_block]

    return required_blocks
