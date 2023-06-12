from basecontexts import BaseContexts


class FlaskDebug(BaseContexts):
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
        self.postordered_nodes = []

    def post_order_traverse(self, root_node, program_content):
        """
        This functions returns postorder traversal of nodes and corresponding
        node literals.
        Args:
            root_node: root node of tree_sitter tree
            program_content: Program in string format from which we need to
                extract node literal
        Returns:
            None
        """
        if(len(root_node.children) == 0):
            literal = bytes(program_content, "utf8")[
                root_node.start_byte:root_node.end_byte
            ].decode("utf8")
            self.postordered_nodes.append((literal, root_node))
        else:
            for ch in root_node.children:
                self.post_order_traverse(ch, program_content)

    def get_app_defination_line(self, root_node, program_content):
        """
        This functions returns the line where node which contains Flask app defination.
        Args:
            root_node: root node of tree_sitter tree
            program_content: Program in string format from which we need to
                extract node literal.
        Returns:
            tree_sitter node containing start_line to end_line
        """
        self.post_order_traverse(root_node, program_content)

        app_index = (-1, -1)
        for i, node in enumerate(self.postordered_nodes):
            # no need to check IndexError as CodeQL raises this flag
            # only if a flask app is running
            if(node[0] == 'Flask'
                    and self.postordered_nodes[i + 1][0] == '('
                    and self.postordered_nodes[i + 2][0] == '__name__'
                    and self.postordered_nodes[i + 3][0] == ')'):
                app_index = (node[1].start_point[0], node[1].end_point[0])
                break

        return app_index


def get_query_specific_context(program_content, parser, file_pah, message, result_span, aux_result_df=None):
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

    context_object = FlaskDebug(parser)
    all_blocks = context_object.get_all_blocks(program_content)

    tree = parser.parse(bytes(program_content, "utf8"))
    root_node = tree.root_node

    # app definition indices
    required_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)

    flask_def_start, flask_def_end = context_object.get_app_defination_line(root_node, program_content)
    for block in all_blocks:
        block_start_line = block.start_line
        block_end_line = block.end_line
        if(len(block.other_lines) == 0):
            contains_def = (flask_def_start >= block_start_line and flask_def_end <= block_end_line)
            # start_line and end_line marks use of debug mode
            contains_debug = (start_line >= block_start_line and end_line <= block_end_line)
            if(contains_def or contains_debug):
                block.relevant = True
        else:
            block_specific_lines = block.other_lines
            contains_def = (flask_def_start in block_specific_lines and flask_def_end in block_specific_lines)
            contains_debug = (start_line in block_specific_lines and end_line in block_specific_lines)
            if(contains_def or contains_debug):
                block.relevant = True

        # to avoid adding same Block twice, in case
        # app defination and debug mode belongs to one Block
        if(block.relevant and (block not in required_blocks)):
            required_blocks.append(block)

    # Add local_block, if not present
    local_block.relevant = True
    if(local_block not in required_blocks):
        required_blocks.append(local_block)

    return required_blocks
