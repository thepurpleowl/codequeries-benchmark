from basecontexts import BaseContexts
from basecontexts import MODULE_FUNCTION, CLASS_FUNCTION
from get_context import __columns__
from collections import namedtuple
import re

Call = namedtuple('Call', 'func_name start_line start_col end_line end_col')


class WrongNumberArgumentsInCall(BaseContexts):
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

    def create_usable_func_map(self, message, start_line, start_column, end_line, end_column):
        """
        Args:
            message: CodeQL result message
            start_line: start_line of auxillary span
            start_column: start_column of auxillary span
            end_line: end_line of auxillary span
            end_column: end_column of auxillary span
        Returns:
            Fully qualified function name
        """
        matches = re.findall(r"\[\[""(.*)""\\|", message)
        qualified_func = matches[0].strip('"')

        # CodeQL has 1-based index
        return Call(qualified_func, int(start_line) - 1, int(start_column) - 1,
                    int(end_line) - 1, int(end_column))

    def get_used_functions(self, file_path, aux_result_df):
        """
        This functions returns relevant Blocks as query specific context.
        Args:
            file_path : file of program_content
        aux_result_df:  auxiliary query results dataframe
        Returns:
            A list consisting lines using import functionality
        """
        used_import_df = aux_result_df.loc[(aux_result_df["Name"] == 'Function call map') & (aux_result_df["Path"] == file_path),
                                           __columns__]
        if(used_import_df.shape[0] == 0):
            return set()

        # Get lines in start and end line, with
        # consideration that CodeQL has 1-based index
        used_import_df["call_map"] = (used_import_df.apply(
                                      lambda x: self.create_usable_func_map(x.Message,
                                                                            x.Start_line, x.Start_column,
                                                                            x.End_line, x.End_column),
                                      axis=1))

        used_func_calls = used_import_df["call_map"].tolist()
        return used_func_calls


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
    start_col = result_span.start_col
    end_line = result_span.end_line
    end_col = result_span.end_col

    context_object = WrongNumberArgumentsInCall(parser)

    # local_block is relevant
    local_block = context_object.get_local_block(program_content, start_line, end_line)
    local_block.relevant = True
    required_blocks = [local_block]

    local_block_lines = local_block.other_lines
    if(not local_block_lines):
        local_block_lines = [i for i in range(local_block.start_line, local_block.end_line + 1)]

    file_path = '/' + file_path
    used_func_calls = context_object.get_used_functions(file_path, aux_result_df)
    relevant_call_maps = []
    for call_map in used_func_calls:
        if((call_map.start_line, call_map.start_col, call_map.end_line, call_map.end_col)
                == (start_line, start_col, end_line, end_col)):
            relevant_call_maps.append(call_map)

    all_blocks = context_object.get_all_blocks(program_content)
    for block in all_blocks:
        add_block = False
        if(block.block_type == CLASS_FUNCTION):
            block_func_name = block.metadata.split('.')[-1]
            block_class_name = block.metadata.split('.')[-2]

            for call_map in used_func_calls:
                used_func_name = call_map[0].split('.')[-1]
                used_func_class = call_map[0].split('.')[0]  # enclosing class in case of inner class
                if(block_func_name == used_func_name
                        and block_class_name == used_func_class
                        and (call_map.start_line in local_block_lines
                             or call_map.end_line in local_block_lines)):
                    add_block = True
                    # if specific used call is relevant
                    if(call_map in relevant_call_maps):
                        block.relevant = True

        elif(block.block_type == MODULE_FUNCTION):
            block_func_name = block.metadata.split('.')[-1]

            for call_map in used_func_calls:
                used_func_name = call_map[0]
                if(block_func_name == used_func_name
                        and (call_map.start_line in local_block_lines
                             or call_map.end_line in local_block_lines)):
                    add_block = True
                    # if specific used call is relevant
                    if(call_map in relevant_call_maps):
                        block.relevant = True

        # If already not in required_blocks and add_block == True
        # then add block to required_blocks. This condition is
        # required to avoid duplicate local_block
        if(add_block and (block not in required_blocks)):
            required_blocks.append(block)

    return required_blocks
