from basecontexts import BaseContexts, MODULE_OTHER
from get_context import __columns__


class UnusedImport(BaseContexts):
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

    def get_import_use_lines(self, file_path, aux_result_df):
        """
        This functions returns relevant Blocks as query specific context.
        Args:
            file_path : file of program_content
            aux_result_df:  auxiliary query results dataframe
        Returns:
            A list consisting lines using import functionality
        """
        used_import_df = aux_result_df.loc[(aux_result_df["Name"] == 'Used import') & (aux_result_df["Path"] == file_path),
                                           __columns__]
        if(used_import_df.shape[0] == 0):
            return set()

        # Get lines in start and end line, with
        # consideration that CodeQL has 1-based index
        used_import_df["Lines"] = (used_import_df.apply(
                                   lambda x: [i for i in range(int(x.Start_line) - 1, int(x.End_line))],
                                   axis=1))

        lines_using_import = set([line for line_sublist in used_import_df["Lines"].tolist()
                                  for line in line_sublist])

        return lines_using_import


def get_query_specific_context(program_content, parser, file_path, message, result_span, aux_result_df):
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

    context_object = UnusedImport(parser)
    all_blocks = context_object.get_all_blocks(program_content)

    file_path = '/' + file_path
    lines_using_import = context_object.get_import_use_lines(file_path, aux_result_df)

    relevant_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)
    for block in all_blocks:
        # MODULE_OTHER always relevant
        if(block.block_type == MODULE_OTHER):
            block.relevant = True
        elif(block == local_block):
            block.relevant = True
        else:
            block_lines = context_object.get_block_lines(block)
            for line in block_lines:
                if(line in lines_using_import):
                    block.relevant = True

        relevant_blocks.append(block)

    return relevant_blocks
