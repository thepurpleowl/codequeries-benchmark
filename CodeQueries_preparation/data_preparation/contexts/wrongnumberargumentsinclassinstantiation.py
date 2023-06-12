from basecontexts import BaseContexts, CLASS_FUNCTION, CALL_NODE_TYPE
from basecontexts import ContextRetrievalError
import re


class WrongNumberArgumentsInClassInstantiation(BaseContexts):
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
        self.target_node = None

    def set_target_node(self, root_node, start_line, end_line, start_col, end_col):
        """
        This functions returns the node which contains start_line to end_line
        in the tree_sitter tree.
        Args:
            root_node: root node of tree_sitter tree
            start_line : start line of a span
            end_line : end line of a span
            start_col : start column of answer span
            end_col : end column of answer span
        Returns:
            tree_sitter node containing start_line to end_line
        """
        if(root_node.type == CALL_NODE_TYPE
                and root_node.start_point[0] == start_line
                and root_node.end_point[0] == end_line
                and root_node.start_point[1] == start_col
                and root_node.end_point[1] == end_col):
            self.target_node = root_node
            return
        else:
            for ch in root_node.children:
                self.set_target_node(ch, start_line, end_line, start_col, end_col)

        return

    def postorder_traverse(self, root_node, program_content):
        """
        This functions returns postorder traversal of nodes and corresponding
        node literals.
        Args:
            root_node: root node of tree_sitter tree
            program_content: Program in string format from which we need to
                extract node literal.
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
                self.postorder_traverse(ch, program_content)

    def get_target_class_name(self, root_node, program_content, start_line, end_line, start_col, end_col):
        """
        This functions returns class name which is incorrectly instantiated.
        Args:
            root_node: root node of tree_sitter tree
            program_content: Program in string format from which we need to
                extract node literal
            start_line : start line of a span
            end_line : end line of a span
            start_col : start column of answer span
            end_col : end column of answer span
        Returns:
            None
        """
        self.set_target_node(root_node, start_line, end_line, start_col, end_col)
        self.postorder_traverse(self.target_node, program_content)
        for i, node in enumerate(self.postordered_nodes):
            if(node[0] == '('):
                class_name = self.postordered_nodes[i - 1][0]
                break

        return class_name


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
    # start_col = result_span.start_col
    end_line = result_span.end_line
    # end_col = result_span.end_col

    context_object = WrongNumberArgumentsInClassInstantiation(parser)

    # tree = parser.parse(bytes(program_content, "utf8"))
    # root_node = tree.root_node

    # class instantiation Block is relevant
    local_block = context_object.get_local_block(program_content, start_line, end_line)
    local_block.relevant = True
    required_blocks = [local_block]

    # class_name = context_object.get_target_class_name(root_node, program_content,
    #                                                   start_line, end_line,
    #                                                   start_col, end_col)

    matches = re.findall(r"relative:\/\/\/[a-zA-Z0-9_.]*:(\d+):(\d+):(\d+):(\d+)", message)
    # get the class of class_name
    req_class_block = None
    if len(matches) == 1:
        # len shld be always one in this query
        # if len==0, then some inbuilt class call
        all_classes = context_object.get_all_classes(program_content)
        init_start_line = int(matches[0][0])
        init_end_line = int(matches[0][2])

        for block in all_classes:
            block_start_line = block.start_line
            block_end_line = block.end_line
            if (block_start_line <= init_start_line
                    and block_end_line >= init_end_line):
                req_class_block = block
                break

    if(len(matches) == 1 and req_class_block is None):
        raise ContextRetrievalError({"message": "No __init__ in class",
                                    "type": "Wrong number of arg in class instantiation"})

    # if class is not from current module (OR) an inner class
    # (OR) couldn't be found
    if(req_class_block is not None):
        # Blocks from corresponding class
        all_blocks = context_object.get_all_blocks(program_content)
        for block in all_blocks:
            if(block.start_line >= req_class_block.start_line
                    and block.end_line <= req_class_block.end_line):
                if(block.block_type == CLASS_FUNCTION):
                    func_name = block.metadata.split('.')[-1]
                    if(func_name == '__init__'):
                        block.relevant = True
                required_blocks.append(block)

    return required_blocks
