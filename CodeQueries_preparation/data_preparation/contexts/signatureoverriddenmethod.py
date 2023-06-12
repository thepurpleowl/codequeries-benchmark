from basecontexts import BaseContexts, CLASS_FUNCTION
import math


class SignatureOverriddenMethod(BaseContexts):
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

    def get_mro_index(self, child_class_mro, class_name):
        """
        This function checks if specific parent CLASS FUNCTION Block
        should be marked relevant
        Args:
            child_class_mro : mro of flagged class
            class_name : target class
        Returns:
            Index of class_name in child_class_mro
        """
        class_mro_index = -1
        for i, cls in enumerate(child_class_mro):
            if(cls == class_name):
                class_mro_index = i
        return class_mro_index

    def is_parent_block_relevant(self, program_content, child_class, target_func_name, target_func_class, parent_class_blocks):
        """
        This function checks if specific parent CLASS FUNCTION Block
        should be marked relevant
        Args:
            program_content: Program in string format from which we need to
            extract classes
            target_func_name : CLASS_FUNCTION Block function name
            target_class_name : CLASS_FUNCTION Block class name
            child_class : flagged class name
            parent_class_blocks : potential relevant parent class Blocks
        Returns:
            A Boolean value denoting whether to add or not
        """
        child_class_mro = self.get_class_MRO(program_content, child_class)

        func_class_index = self.get_mro_index(child_class_mro, target_func_class)
        if(func_class_index == -1):
            return False

        relevant_class_index = math.inf
        mro_func_class = None
        for block in parent_class_blocks:
            if(block.block_type == CLASS_FUNCTION):
                func_name = block.metadata.split('.')[-1]
                class_name = block.metadata.split('.')[-2]
                class_index = self.get_mro_index(child_class_mro, class_name)
                if(func_name == target_func_name
                        and (class_index != -1)
                        and (class_index < relevant_class_index)):
                    relevant_class_index = class_index
                    mro_func_class = class_name
        return (mro_func_class == target_func_class)


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

    context_object = SignatureOverriddenMethod(parser)
    local_class_block, parent_class_blocks = context_object.get_local_and_super_class(program_content, start_line, end_line)
    all_blocks = context_object.get_all_blocks(program_content)

    required_blocks = []
    local_block = context_object.get_local_block(program_content, start_line, end_line)

    # for intersecting function check
    child_class_functions = set()
    parent_class_functions = dict()  # dictionary of set

    required_child_class_blocks = []
    required_parent_class_blocks = []
    if local_class_block is not None:
        for block in all_blocks:
            # if a block in local class
            if (block.start_line >= local_class_block.start_line
                    and block.end_line <= local_class_block.end_line):
                if(block == local_block):
                    block.relevant = True
                required_child_class_blocks.append(block)
                # for intersecting function check
                if(block.block_type == CLASS_FUNCTION):
                    child_class_functions.add(block.metadata.split('.')[-1])
            # if a block in any of the parent class
            else:
                for p_block in parent_class_blocks:
                    class_name = p_block.metadata.split('.')[-1]
                    if (block.start_line >= p_block.start_line
                            and block.end_line <= p_block.end_line):
                        required_parent_class_blocks.append(block)
                        # for intersecting function check
                        if(block.block_type == CLASS_FUNCTION):
                            func_name = block.metadata.split('.')[-1]
                            if(class_name in parent_class_functions):
                                parent_class_functions[class_name].add(func_name)
                            else:
                                parent_class_functions[class_name] = set([func_name])

        # functions in parent blocks that would require MRO
        mro_functions = dict()
        if(child_class_functions):
            for cls, func_set in parent_class_functions.items():
                mro_functions[cls] = child_class_functions - func_set

        for cls, func_set in mro_functions.items():
            for func in func_set:
                mro_func_block = context_object.get_mro_function_block(func, cls, program_content)
                if(mro_func_block is not None
                        and mro_func_block not in required_parent_class_blocks):
                    # Blocks retrieved with MRO are not relevant
                    # because of declaredAttribute
                    required_parent_class_blocks.append(mro_func_block)

        # mark relevance
        for block in required_child_class_blocks:
            for p_block in required_parent_class_blocks:
                if(block.block_type == CLASS_FUNCTION and p_block.block_type == CLASS_FUNCTION):
                    overriding_func_name = block.metadata.split('.')[-1]
                    child_class = block.metadata.split('.')[-2]

                    overridden_func_name = p_block.metadata.split('.')[-1]
                    overridden_func_class = p_block.metadata.split('.')[-2]
                    if(overriding_func_name == overridden_func_name
                            and context_object.is_parent_block_relevant(program_content, child_class,
                                                                        overridden_func_name, overridden_func_class,
                                                                        required_parent_class_blocks)):
                        block.relevant = True
                        p_block.relevant = True

        required_blocks.extend(required_child_class_blocks)
        required_blocks.extend(required_parent_class_blocks)

    return required_blocks
