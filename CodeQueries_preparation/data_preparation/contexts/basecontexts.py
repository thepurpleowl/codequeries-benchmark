from dataclasses import dataclass
from typing import Dict, List, Tuple
import re
import sys
sys.path.insert(0, './contexts')
import get_mro
from get_builtin_stub import get_custom_stub

# defined Block block_type
MODULE_FUNCTION = 'MODULE_FUNCTION'
CLASS_FUNCTION = 'CLASS_FUNCTION'
CLASS_OTHER = 'CLASS_OTHER'
MODULE_OTHER = 'MODULE_OTHER'
STUB = 'STUB'
IMPORT = 'IMPORT'  # not reauired to be added in proto def
# utility types
ROOT_BLOCK_TYPE = 'root'
CLASS_BLOCK_TYPE = 'class'
# tree-sitter node types
CALL_NODE_TYPE = 'call'
IMPORT_NODE_TYPE = ['import_statement', 'import_from_statement']
DECORATOR_NODE_TYPE = 'decorator'
IDENTIFIER_NODE_TYPE = 'identifier'
# python inbuilt datatypes
INBUILT_TYPES = ['list', 'dict', 'set', 'frozenset', 'tuple',
                 'bool', 'int', 'float', 'complex', 'str', 'NoneType', 'None']
DEFAULT_BLOCK_HEADER = 'module'


@dataclass
class Block:
    """
    Dataclass used to define Block data strcuture
    ...

    Attributes
    ----------
    start_line : str
        start line of a Block
    end_line : str
        end line of a Block
    other_lines : list
        non-continuous block of lines in Block of type CLASS_OTHER and MODULE_OTHER
    content : int
        Block content
    metadata : str
        any relevant metadata
    block_type : str
        one of MODULE_FUNCTION, CLASS_FUNCTION, CLASS_OTHER, MODULE_OTHER
    relevant : bool
        relevance of a block wrt to a query
    block_header: str
        additional class definition/auxillary block header
    mro: list
        class MRO(for CLASS_FUNCTION blocks only)
    """
    start_line: int
    end_line: int
    other_lines: List[int]
    content: str
    metadata: str = ''
    block_type: str = ''
    relevant: bool = False
    block_header: str = ''
    mro: Tuple[str, ...] = ('__', '__class__')


class ContextRetrievalError(Exception):
    """
    Raise context retrieval specific errors
    message:
        error message
    type:
        generic: errors during retrieval logic
        empty_context: no context is retrieved
    """


class BaseContexts:
    """
    This is the base class for methods used by different queries to extract
    contexts and corresponding metadata from program content.
    Metadata contains start/end line number, location wrt the module for
    corresponding context Block.

    For now, inner class/methods will be part of surrounding class/method.
    """

    def __init__(self, parser):
        """
        Args:
            parser: Tree sitter parser object
        """
        self.tree_sitter_parser = parser
        self.FUNCTION_DEFINITION = "function_definition"
        self.CLASS_DEFINITION = "class_definition"
        self.DECORATED_DEFINITION = "decorated_definition"
        self.BLOCK_INVARIANT_TYPES = ['if_statement', 'elif_clause', 'else_clause',
                                      'try_statement', 'except_clause', 'with_statement',
                                      'if', 'elif', 'else',
                                      'try', 'except', 'with']
        self.BLOCK_DEFINITION = "block"
        self.START_LINE = "start_line"
        self.END_LINE = "end_line"
        self.IMPORT_STATEMENTS = []

    def get_function_name(self, node, program_content):
        """
        This function returns name of function node.
        Args:
            node: tree_sitter node
            program_content: Program in string format from which we need to
            extract functions
        Returns:
            Name of the corresponding function
        """
        if(node.type == self.FUNCTION_DEFINITION):
            func_name = (
                bytes(program_content, "utf8")[
                    node.children[1].start_byte: node.children[1].end_byte
                ].decode("utf8")
            )
        else:
            for child in node.children:
                if child.type == self.FUNCTION_DEFINITION:
                    func_name = (
                        bytes(program_content, "utf8")[
                            child.children[1].start_byte: child.children[1].end_byte
                        ].decode("utf8")
                    )

        return func_name

    def get_class_name(self, node, program_content):
        """
        This function returns name of class node.
        Args:
            node: tree_sitter node
            program_content: Program in string format from which we need to
            extract functions
        Returns:
            Name of the corresponding class
        """
        if(node.type == self.CLASS_DEFINITION):
            class_name = (
                bytes(program_content, "utf8")[
                    node.children[1].start_byte: node.children[1].end_byte
                ].decode("utf8")
            )
        else:
            for child in node.children:
                if child.type == self.CLASS_DEFINITION:
                    class_name = (
                        bytes(program_content, "utf8")[
                            child.children[1].start_byte: child.children[1].end_byte
                        ].decode("utf8")
                    )

        return class_name

    def is_class_node(self, node):
        """
        This function checks if node is a class node. If not returns None.
        Args:
            node: tree_sitter node
        Returns:
            node if it's a class node, None otherwise
        """
        class_node = None

        if (node.type == self.CLASS_DEFINITION):
            class_node = node
        # class with single/multiple decorator
        elif(node.type == self.DECORATED_DEFINITION
                and node.children[-1].type == self.CLASS_DEFINITION):
            class_node = node.children[-1]

        return class_node

    def is_function_node(self, node):
        """
        This function checks if node is a function node.
        Args:
            node: tree_sitter node
        Returns:
            Return boolean value denoting whether the is a function node.
        """
        is_function = False

        if (node.type == self.FUNCTION_DEFINITION
            # function with single/multiple decorator
            or (node.type == self.DECORATED_DEFINITION
                and node.children[-1].type == self.FUNCTION_DEFINITION)):
            is_function = True

        return is_function

    def get_invariant_enclosed_functions(self, root_node):
        """
        This function returns functions present as children of a tree-sitter node
        at module level of type BLOCK_INVARIANT_TYPES.
        Args:
            root_node: tree_sitter node at module level
        Returns:
            Return list of function nodes under root_node
        """
        invariant_enclosed_function_nodes = []

        for node in root_node.children:
            if(self.is_function_node(node)):
                invariant_enclosed_function_nodes.append(node)
            elif(node.type in self.BLOCK_INVARIANT_TYPES
                    or node.type == self.BLOCK_DEFINITION):
                inner_nodes = self.get_invariant_enclosed_functions(node)
                for i_node in inner_nodes:
                    invariant_enclosed_function_nodes.append(i_node)

        return invariant_enclosed_function_nodes

    def get_invariant_enclosed_classes(self, root_node):
        """
        This function returns classes present as children of a tree-sitter node
        at module level of type BLOCK_INVARIANT_TYPES.
        Args:
            root_node: tree_sitter node at module level
        Returns:
            Return list of class nodes under root_node
        """
        invariant_enclosed_class_nodes = []

        for node in root_node.children:
            if(self.is_class_node(node) is not None):
                invariant_enclosed_class_nodes.append(node)
            elif(node.type in self.BLOCK_INVARIANT_TYPES
                    or node.type == self.BLOCK_DEFINITION):
                inner_nodes = self.get_invariant_enclosed_classes(node)
                for i_node in inner_nodes:
                    invariant_enclosed_class_nodes.append(i_node)

        return invariant_enclosed_class_nodes

    def get_class_bases(self, block_header: str):
        """
        This function return base classes from the class definition string.
        Args:
            block_header: class definition string
        Returns:
            A tuple of classes representing MRO of block_header
        """
        parent_class_mro = []
        parent_classes = []
        # if inherits some base class
        if('(' in block_header
                and '):' in block_header):
            parent_classes = (block_header.split('(')[-1].strip().
                              split('):')[0].strip()).split(',')
        for base_class in parent_classes:
            temp_class = base_class.strip()
            if(temp_class):
                parent_class_mro.append(temp_class)

        return parent_class_mro

    def get_class_MRO(self, program_content: str, cls: str):
        """
        This function return class MRO from the class definition string
        Args:
            program_content: Program in string format from which we need to
            extract Blocks
            cls: target class name
        Returns:
            A tuple of classes representing MRO of block_header
        """
        all_classes = self.get_all_classes(program_content)
        class_bases_dict: Dict[str, List[str]] = {'object': []}
        for class_block in all_classes:
            class_name = class_block.metadata.split('.')[-1]
            class_bases_dict[class_name] = self.get_class_bases(class_block.block_header)

        try:
            class_mro = get_mro.mro(class_bases_dict, cls)
            if(class_mro[-1] == 'object'):
                # first element is the class itself
                class_mro = class_mro[:-1]
        except RecursionError:
            class_mro = [cls]
        except Exception:
            class_mro = [cls]
        return tuple(class_mro)

    def get_block_header(self, program_content, class_node):
        """
        This function return class header from the class definition string.
        Args:
            progam_content: source code
            class_node: tree_sitter class node
        Returns:
            Class header from class_node
        """
        f = (
            bytes(program_content, "utf8")[
                # from start of class definition till ':'
                class_node.start_byte: class_node.children[-2].end_byte
            ].decode("utf8")
        )

        return f

    def get_all_functions(self, program_content: str, k: int, parent: str, depth: int,
                          block_header: str = DEFAULT_BLOCK_HEADER, class_MRO: Tuple[str, ...] = ('__', '__class__')):
        """
        This function extracts all functions.
        Args:
            program_content: Program in string format from which we need to
            extract functions
            k: Start line
            parent: parent node of a node used to store metadata
            depth: depth of tree_sitter tree to handle inner classes
            block_header: class definition to be used with respective CLASS_FUNCTION
        Returns:
            A list of Blocks of type MODULE_FUNCTION or CLASS_FUNCTION
        """
        function_blocks = []

        # CAUTION: if and when inner class implementation will be done,
        # lines shouldn't be part of recursion stack
        lines = program_content.split('\n')

        tree = self.tree_sitter_parser.parse(bytes(program_content, "utf8"))
        root_node = tree.root_node
        children = root_node.children

        for i in range(len(children)):
            # children[i] is a function
            if(self.is_function_node(children[i])):
                f = (
                    bytes(program_content, "utf8")[
                        children[i].start_byte: children[i].end_byte
                    ].decode("utf8")
                )
                f_start = children[i].start_point[0] + k
                f_end = children[i].end_point[0] + k
                func_name = self.get_function_name(children[i], program_content)
                f_block = Block(f_start, f_end, [], f, parent + '.' + func_name,
                                MODULE_FUNCTION, False, block_header, class_MRO)
                function_blocks.append(f_block)
            # children[i] is a class
            elif(self.is_class_node(children[i]) is not None):
                if(depth > 0):
                    continue
                class_node = self.is_class_node(children[i])
                line_no = class_node.children[2].end_point[0] + k
                line_no_end = class_node.end_point[0] + k
                class_name = self.get_class_name(children[i], program_content)
                temp_block_header = self.get_block_header(program_content, class_node)
                temp_class_MRO = self.get_class_MRO(program_content, class_name)
                current_class_MRO = class_MRO + temp_class_MRO

                f_a_m = self.get_all_functions(
                    '\n'.join(i for i in lines[line_no + 1:line_no_end + 1]),
                    line_no + 1,
                    parent + '.' + class_name, depth + 1,
                    temp_block_header, current_class_MRO
                )
                for func in f_a_m:
                    func.block_type = CLASS_FUNCTION
                function_blocks.extend(f_a_m)
            # children[i] is of BLOCK_INVARIANT_TYPES
            elif(children[i].type in self.BLOCK_INVARIANT_TYPES
                    or children[i].type == self.BLOCK_DEFINITION):
                invariant_enclosed_function_nodes = self.get_invariant_enclosed_functions(children[i])
                invariant_enclosed_class_nodes = self.get_invariant_enclosed_classes(children[i])

                # add function nodes found under children[i]
                for func_node in invariant_enclosed_function_nodes:
                    f = (
                        bytes(program_content, "utf8")[
                            func_node.start_byte: func_node.end_byte
                        ].decode("utf8")
                    )
                    f_start = func_node.start_point[0] + k
                    f_end = func_node.end_point[0] + k
                    func_name = self.get_function_name(func_node, program_content)
                    f_block = Block(f_start, f_end, [], f, parent + '.' + func_name,
                                    MODULE_FUNCTION, False, block_header, class_MRO)
                    function_blocks.append(f_block)

                # add function nodes present in classes found under children[i]
                for class_node in invariant_enclosed_class_nodes:
                    if(depth > 0):
                        continue
                    class_node = self.is_class_node(class_node)
                    line_no = class_node.children[2].end_point[0] + k
                    line_no_end = class_node.end_point[0] + k
                    class_name = self.get_class_name(class_node, program_content)
                    temp_block_header = self.get_block_header(program_content, class_node)
                    temp_class_MRO = self.get_class_MRO(program_content, class_name)
                    current_class_MRO = class_MRO + temp_class_MRO

                    f_a_m = self.get_all_functions(
                        '\n'.join(i for i in lines[line_no + 1:line_no_end + 1]),
                        line_no + 1,
                        parent + '.' + class_name, depth,
                        temp_block_header, current_class_MRO
                    )
                    for func in f_a_m:
                        func.block_type = CLASS_FUNCTION
                    function_blocks.extend(f_a_m)

        return function_blocks

    def get_all_classes(self, program_content: str):
        """
        This functions extracts all classes.
        Args:
            program_content: Program in string format from which we need to
            extract classes.
        Returns:
            List of Blocks of class data.
        """
        def_class_MRO: Tuple[str, ...] = ('__', '__class__')
        class_blocks = []

        tree = self.tree_sitter_parser.parse(bytes(program_content, "utf8"))
        root_node = tree.root_node
        children = root_node.children

        for i in range(len(children)):
            # children[i] is a class
            if (self.is_class_node(children[i]) is not None):
                class_node = self.is_class_node(children[i])
                class_start_line = children[i].start_point[0]
                class_end_line = children[i].end_point[0]
                class_name = self.get_class_name(children[i], program_content)
                block_header = self.get_block_header(program_content, class_node)
                current_class_MRO = def_class_MRO + tuple(self.get_class_bases(block_header))
                class_content = bytes(program_content, "utf8")[
                    children[i].start_byte:children[i].end_byte
                ].decode("utf8")

                class_blocks.append(Block(class_start_line,
                                          class_end_line,
                                          [],
                                          class_content,
                                          ROOT_BLOCK_TYPE + '.' + class_name,
                                          CLASS_BLOCK_TYPE,
                                          False,
                                          block_header,
                                          current_class_MRO))
            # children[i] is of BLOCK_INVARIANT_TYPES
            elif(children[i].type in self.BLOCK_INVARIANT_TYPES
                    or children[i].type == self.BLOCK_DEFINITION):
                invariant_enclosed_class_nodes = self.get_invariant_enclosed_classes(children[i])

                # add class nodes found under children[i]
                for class_node in invariant_enclosed_class_nodes:
                    class_start_line = class_node.start_point[0]
                    class_end_line = class_node.end_point[0]
                    class_name = self.get_class_name(class_node, program_content)
                    block_header = self.get_block_header(program_content, class_node)
                    current_class_MRO = def_class_MRO + tuple(self.get_class_bases(block_header))
                    class_content = bytes(program_content, "utf8")[
                        class_node.start_byte:class_node.end_byte
                    ].decode("utf8")

                    class_blocks.append(Block(class_start_line,
                                              class_end_line,
                                              [],
                                              class_content,
                                              ROOT_BLOCK_TYPE + '.' + class_name,
                                              CLASS_BLOCK_TYPE,
                                              False,
                                              block_header,
                                              current_class_MRO))

        return class_blocks

    def get_all_blocks(self, program_content: str, content_type=None):
        """
        This functions extracts four type of Blocks except STUB type.
        Args:
            program_content: Program in string format from which we need to
            extract Blocks
            content_type: STUB or not, to correctly produce block_header for
                          stub content
        Returns:
            A list of possible Blocks in program_content

            The list contains Blocks in the order - (MODULE_FUNCTION/CLASS_FUNCTION),
            CLASS_OTHER and MODULE_OTHER. This order is used in distributable queries.
        """
        lines = program_content.split('\n')
        tree = self.tree_sitter_parser.parse(bytes(program_content, "utf8"))
        root_node = tree.root_node

        # initialize all_blocks with function blocks and
        # block_lines with function block lines
        all_blocks = self.get_all_functions(program_content, 0, ROOT_BLOCK_TYPE, 0)
        block_lines = []
        for block in all_blocks:
            block_lines.extend([i for i in range(block.start_line, block.end_line + 1)])

        # line numbers that are not covered by any function Blocks
        left_out_lines = [i for i in range(root_node.end_point[0] + 1) if i not in block_lines]

        # initialize `left_out_class_lines` for line numbers in
        # some classese that are not covered by any function Blocks.
        # Get such lines and update `left_out_class_lines` and create such Blocks.
        left_out_class_lines = []
        all_classes = self.get_all_classes(program_content)
        for class_block in all_classes:
            class_start_line = class_block.start_line
            class_end_line = class_block.end_line
            class_MRO = class_block.mro
            class_name = class_block.metadata.split('.')[-1]

            class_specific_block_lines = []
            for j in left_out_lines:
                if(j >= class_start_line and j <= class_end_line):
                    class_specific_block_lines.append(j)
            left_out_class_lines.extend(class_specific_block_lines)

            class_specific_block_content = \
                '\n'.join(lines[j] for j in class_specific_block_lines)

            all_blocks.append(Block(class_start_line,
                                    class_end_line,
                                    class_specific_block_lines,
                                    class_specific_block_content,
                                    ROOT_BLOCK_TYPE + '.' + class_name,
                                    CLASS_OTHER,
                                    False,
                                    DEFAULT_BLOCK_HEADER,
                                    class_MRO))

        # remaining left out lines are statements at module level
        remaining_lines = [i for i in left_out_lines if i not in left_out_class_lines]
        if(len(remaining_lines) > 0):
            remaining_start_line = root_node.start_point[0]
            remaining_end_line = root_node.end_point[0]

            remaining_block_content = \
                '\n'.join(lines[j] for j in remaining_lines)

            all_blocks.append(Block(remaining_start_line,
                                    remaining_end_line,
                                    remaining_lines,
                                    remaining_block_content,
                                    ROOT_BLOCK_TYPE,
                                    MODULE_OTHER,
                                    False,
                                    DEFAULT_BLOCK_HEADER))

        if(content_type == STUB):
            for block in all_blocks:
                # MODULE_OTHER/MODULE_FUNCTION not possible
                block.block_header = STUB + ' ' + block.block_header

        return all_blocks

    def preprocess_import(self, statement):
        """
        This function returns preprocessed import statement.
        Preprocess step ensures single import in a line.
        Args:
            statement: input import statement
        Returns:
            preprocessed import statement
        """
        statement = statement.replace('\\', '')
        statement = statement.replace('\n', '')
        module_group = (statement.split('as ')[0].strip().
                        split('import ')[1].strip())
        module_names = [module.strip() for module in module_group.split(',')]
        preprocessed_statement = []
        for module in module_names:
            preprocessed_statement.append(
                statement.split('import ')[0] + 'import ' + module)

        return preprocessed_statement

    def set_imports(self, program_content, root_node):
        """
        This function extracts all import statements in program_content
        and stores them for further processing.
        Args:
            program_content: Program in string format
            root_node: tree_sitter node at module level
        Returns:
            None
        """
        children = root_node.children
        if(len(children) == 0):
            return
        for node in children:
            if(node.type in IMPORT_NODE_TYPE):
                statement = bytes(program_content, "utf8")[
                    node.start_byte:node.end_byte
                ].decode("utf8")

                preprocessed_import = self.preprocess_import(statement)
                statement_start = node.start_point[0]
                statement_end = node.end_point[0]
                for line in preprocessed_import:
                    if line not in self.IMPORT_STATEMENTS:
                        self.IMPORT_STATEMENTS.append((line,
                                                      (statement_start, statement_end)))
            else:
                self.set_imports(program_content, node)

    def get_import_block(self, program_content):
        """
        This function extracts import statements.
        Args:
            program_content: Program in string format
        Returns:
            An auxillary Block, which may/may not be part of input
            program_content with all preprocessed import statements.
        """
        tree = self.tree_sitter_parser.parse(bytes(program_content, "utf8"))
        root_node = tree.root_node

        self.set_imports(program_content, root_node)
        import_line_indices = []
        import_content = []
        for import_line in self.IMPORT_STATEMENTS:
            import_content.append(import_line[0])
            import_line_start = import_line[1][0]
            import_line_end = import_line[1][1]
            if(import_line_start == import_line_end):
                import_line_indices.append(import_line_start)
            else:
                import_line_indices.extend([i for i in range(import_line_start, import_line_end + 1)])

        import_block = Block(-1, -1, import_line_indices,
                             '\n'.join(import_content), '', IMPORT)

        return import_block

    def get_module(self, program_content: str):
        """
        This function extracts module context.
        Args:
            program_content: Program in string format from which we need to
            extract module
        Returns:
            A Block with module body and corresponding metadata
        """
        tree = self.tree_sitter_parser.parse(bytes(program_content, "utf8"))
        root_node = tree.root_node

        module_body = (
            bytes(program_content, "utf8")[
                root_node.start_byte: root_node.end_byte
            ].decode("utf8")
        )
        module_start = root_node.start_point[0]
        module_end = root_node.end_point[0]

        module_block = Block(module_start, module_end, [], module_body, '', ROOT_BLOCK_TYPE)

        return module_block

    def get_local_block(self, program_content, start_line, end_line):
        """
        This functions returns the Block which contains start_line to end_line.
        Args:
            program_content: Program in string format from which we need to
            extract classes.
            start_line : start line of a tree_sitter node
            end_line : end line of a tree_sitter node
        Returns:
            Surrounding Block containing start_line to end_line
        """
        all_blocks = self.get_all_blocks(program_content)
        local_block = None
        for block in all_blocks:
            if(start_line >= block.start_line and end_line <= block.end_line):
                if(len(block.other_lines) == 0):
                    local_block = block
                    break
                else:
                    if(start_line in block.other_lines
                       and end_line in block.other_lines):
                        local_block = block
                        break

        return local_block

    def get_transitive_parent_classes(self, parent_class_names, all_class_blocks, k):
        """
        This functions returns the all parent classes by traversing transitively
        on given parent class names and using mro.
        Args:
            parent_class_names: parent class names
            all_class_blocks: all class blocks in a file
        Returns:
            set of all parent class names
        """
        all_parent_class_names = parent_class_names
        new_parent_class_names = []
        for c_block in all_class_blocks:
            if((c_block.metadata.split('.')[-1] in parent_class_names)
                    and (len(c_block.mro) > 2)):
                new_parent_class_names.extend(list(c_block.mro[2:]))

        all_parent_class_names.extend(new_parent_class_names)
        all_parent_class_names = list(set(all_parent_class_names))

        if(len(all_parent_class_names) == k):
            return all_parent_class_names
        else:
            return self.get_transitive_parent_classes(all_parent_class_names, all_class_blocks, len(all_parent_class_names))

    def get_local_and_super_class(self, program_content, start_line, end_line):
        """
        This functions returns the Class Block which contains start_line to end_line
        along with its parent Class Blocks.
        Args:
            program_content: Program in string format from which we need to
            extract classes.
            start_line : start line of a tree_sitter node
            end_line : end line of a tree_sitter node
        Returns:
            Surrounding Class Block containing start_line to end_line along with
            its parent Class Blocks of surrounding class
        """
        class_blocks = self.get_all_classes(program_content)

        local_class_block = None
        for block in class_blocks:
            if(start_line >= block.start_line and end_line <= block.end_line):
                local_class_block = block
                # defined class Blocks are non-overlapping, hence break
                break

        parent_class_blocks = []
        parent_class_names = []
        if local_class_block is not None:
            # first 2 values in class block mro are dummy values
            # for object and self
            if(len(local_class_block.mro) > 2):
                parent_class_names = list(local_class_block.mro[2:])

            parent_class_names = self.get_transitive_parent_classes(parent_class_names, class_blocks, len(parent_class_names))

            # get parent class blocks
            for c_block in class_blocks:
                class_name = c_block.metadata.split('.')[-1]
                if class_name in parent_class_names:
                    # how mro is stored for class blocks inhibits
                    # addition of local class block in parent blocks
                    parent_class_blocks.append(c_block)

        return local_class_block, parent_class_blocks

    def get_transitive_child_classes(self, child_class_names, all_class_blocks, k):
        """
        This functions returns the all child classes by traversing transitively
        on given child class names and using mro.
        Args:
            child_class_names: child class names
            all_class_blocks: all class blocks in a file
        Returns:
            set of all child class names
        """
        all_child_class_names = child_class_names
        new_child_class_names = []
        for c_block in all_class_blocks:
            temp_class_mro = []
            # first 2 values in class block mro are dummy values
            # for object and self
            if(len(c_block.mro) > 2):
                temp_class_mro = c_block.mro[2:]
            for c_name in all_child_class_names:
                if c_name in temp_class_mro:
                    # how mro is stored for class blocks inhibits
                    # addition of local class name in child_class_names
                    c_block_name = c_block.metadata.split('.')[-1]
                    new_child_class_names.append(c_block_name)

        all_child_class_names.extend(new_child_class_names)
        all_child_class_names = list(set(all_child_class_names))

        if(len(all_child_class_names) == k):
            return all_child_class_names
        else:
            return self.get_transitive_child_classes(all_child_class_names, all_class_blocks, len(all_child_class_names))

    def get_local_and_child_class(self, program_content, start_line, end_line):
        """
        This functions returns the Class Block which contains start_line to end_line
        along with child Class Blocks of that class.
        Args:
            program_content: Program in string format from which we need to
            extract classes.
            start_line : start line of a tree_sitter node
            end_line : end line of a tree_sitter node
        Returns:
            Surrounding Class Block containing start_line to end_line along with
            the children Class Blocks of surrounding class
        """
        class_blocks = self.get_all_classes(program_content)
        local_class_block = None
        for block in class_blocks:
            if(start_line >= block.start_line and end_line <= block.end_line):
                local_class_block = block
                # defined class Blocks are non-overlapping, hence break
                break

        child_class_blocks = []
        child_class_names = []
        if local_class_block is not None:
            # get children class names
            local_class_name = local_class_block.metadata.split('.')[-1]
            for c_block in class_blocks:
                temp_class_mro = []
                # first 2 values in class block mro are dummy values
                # for object and self
                if(len(c_block.mro) > 2):
                    temp_class_mro = c_block.mro[2:]
                if local_class_name in temp_class_mro:
                    # how mro is stored for class blocks inhibits
                    # addition of local class name in child_class_names
                    c_block_name = c_block.metadata.split('.')[-1]
                    child_class_names.append(c_block_name)

            child_class_names = self.get_transitive_child_classes(child_class_names, class_blocks, len(child_class_names))

            # get children class blocks
            for c_block in class_blocks:
                class_name = c_block.metadata.split('.')[-1]
                if class_name in child_class_names:
                    child_class_blocks.append(c_block)

        return local_class_block, child_class_blocks

    def get_local_class(self, program_content, start_line, end_line):
        """
        This functions returns the class Block which contains start_line to end_line,
        returns None if not found.
        Args:
            program_content: Program in string format from which we need to
            extract classes
            start_line : start line of a tree_sitter node
            end_line : end line of a tree_sitter node
        Returns:
            Surrounding Class Block containing start_line to end_line
            If no surrounding Class Block then None is returned
        """
        class_blocks = self.get_all_classes(program_content)
        local_class_block = None
        for block in class_blocks:
            if(start_line >= block.start_line and end_line <= block.end_line):
                local_class_block = block
                break

        return local_class_block

    def get_block_lines(self, block):
        """
        This functions returns list of lines in block.
        Args:
            block : a context Block
        Returns:
            list of lines in block
        """
        block_lines = block.other_lines
        if not block_lines:
            block_lines = [i for i in range(block.start_line, block.end_line + 1)]

        return block_lines

    def check_multiple_inheritance_super(self, program_content: str, class_block):
        """
        This functions returns specific super class used for object instantiation
        in class_block content.
        Args:
            program_content: Program in string format from which we need to
            extract classes
            class_block : a Class Block
        Returns:
            specific super class whose __init__() is used, else None
        """
        content = class_block.content
        class_name = class_block.metadata.split('.')[-1]

        class_blocks = self.get_all_blocks(content)
        super_class = None
        for block in class_blocks:
            if(block.block_type == CLASS_FUNCTION):
                function_name = block.metadata.split('.')[-1]
                if(function_name == '__init__'):
                    matches = re.findall(r"(.*)\.__init__\(", content)
                    if(len(matches) == 0):
                        super_class = None
                    elif(len(matches) > 1):
                        # as super() should be resolved to only one class, but `re`
                        # can find >1 if some condition based instantiation is done
                        possible_super_classes: List[str] = list()
                        for match in matches:
                            temp_super_class = (match.split('super')[-1].strip().
                                                strip('(').strip(')').split(',')[0].
                                                strip())
                            if('super' in match.strip()):
                                temp_possible_super_class = self.get_class_MRO(program_content,
                                                                               temp_super_class)[0]
                            else:
                                temp_possible_super_class = temp_super_class
                            possible_super_classes.append(temp_possible_super_class)
                        # as dynamic behavior can't be decided statically,
                        # last occurance of super() is considered to get super_class
                        if(possible_super_classes):
                            super_class = possible_super_classes[-1]
                    else:
                        temp_super_class = (matches[0].strip())
                        if('super' not in temp_super_class):
                            super_class = temp_super_class
                        # else case will be handled with L873-874, where
                        # super_class will be decied by MRO of class_name

        if(super_class is None):
            super_class = self.get_class_MRO(program_content, class_name)[0]
        return super_class

    def get_mro_function_block(self, func_name, current_class, program_content):
        """
        This functions returns specific CLASS_FUNCTION Block with func_name
        if not found in current_class.
        Args:
            func_name: target CLASS_FUNCTION name
            current_class: class for which MRO needed
            program_content: Program in string format from which we need to
            extract Blocks
        Returns:
            specific super class whose __init__() is used
        """
        class_mro = self.get_class_MRO(program_content, current_class)

        all_blocks = self.get_all_blocks(program_content)
        possible_blocks = []
        possible_block_classes = set()
        for block in all_blocks:
            if(block.block_type == CLASS_FUNCTION):
                block_func_name = block.metadata.split('.')[-1]
                if(block_func_name == func_name):
                    block_class_name = block.metadata.split('.')[-2]
                    possible_blocks.append(block)
                    possible_block_classes.add(block_class_name)

        for cls in class_mro:
            if(cls in possible_block_classes):
                for block in possible_blocks:
                    block_func_name = block.metadata.split('.')[-1]
                    block_class_name = block.metadata.split('.')[-2]
                    if(block_func_name == func_name
                            and block_class_name == cls):
                        return block

        # STUB for built-in types
        for cls in class_mro:
            if cls in INBUILT_TYPES:
                mro_stub_content = get_custom_stub(cls, func_name)
                stub_blocks = self.get_all_blocks(mro_stub_content, STUB)
                for block in stub_blocks:
                    block.start_line = -1
                    block.end_line = -1
                    if(block.block_type == CLASS_FUNCTION):
                        block.block_type = STUB
                        block.relevant = True
                        return block

        return None
