from basecontexts import BaseContexts
from basecontexts import CLASS_FUNCTION, MODULE_OTHER, STUB, INBUILT_TYPES
from get_builtin_stub import get_class_specific_call_stub

INBUILT = 'builtin-class'
INCODE = 'class'
IMPORT = 'module-import'


class NonCallableCalled(BaseContexts):
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

    def get_target_class(self, message):
        """
        This functions returns the class type which gave rise to the
        CodeQL flag.
        Args:
            message : CodeQL message
        Returns:
            class_type : one of INBUILT, INCODE, IMPORT
            class_name : class name
        """
        class_type = message.split(':::')[0]
        if(class_type.startswith('class')):
            return INCODE, class_type.split()[-1].strip()
        elif(class_type.startswith('builtin-class')
                and class_type.split()[-1] in INBUILT_TYPES):
            return INBUILT, class_type.split()[-1].strip()
        else:
            return IMPORT, class_type.split()[-1].strip()


def get_query_specific_context(program_content, parser, file_path, message, result_span, aux_result_df=None):
    """
    This functions returns the class name which contains start_line to end_line.
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

    context_object = NonCallableCalled(parser)

    local_block = context_object.get_local_block(program_content, start_line, end_line)
    required_blocks = [local_block]

    '''
        Three class_types
            1.user-defined class
              class DiffieHellman
            2.builtin-class
              (list, dict, set, frozenset, tuple, bool, int, float, complex, str, NoneType/None)
              memoryview, bytearray, bytes
            3.module imports
              built-in method: /babble/babble/include/jython/Lib/test/test_func_jy.py
              builtin-class _io.TextIOWrapper
              builtin-class module
    '''
    class_type, class_name = context_object.get_target_class(message)
    if(class_type == INBUILT):
        class_specific_stub = get_class_specific_call_stub(class_name)
        stub_blocks = context_object.get_all_blocks(class_specific_stub, STUB)
        for block in stub_blocks:
            if(block.block_type == CLASS_FUNCTION
                    and block.metadata.split('.')[-1] == '__call__'):
                block.relevant = True
            block.start_line = -1
            block.end_line = -1
            block.block_type = STUB
            # when deciding for STUB all blocks will
            # be in intermediate context
            required_blocks.append(block)
    else:
        all_blocks = context_object.get_all_blocks(program_content)
        if(class_type == INCODE):
            all_classes = context_object.get_all_classes(program_content)
            cls_start = -1
            cls_end = -1
            for cls in all_classes:
                if(cls.metadata.split('.')[-1] == class_name):
                    cls_start = cls.start_line
                    cls_end = cls.end_line
                    break

            # if the class is inner class
            if(cls_start == -1 and cls_end == -1):
                for block in all_blocks:
                    # check if class is defined in some other type block
                    class_definition = class_type + ' ' + class_name
                    if(class_definition in block.content):
                        if(block in required_blocks):
                            for added_block in required_blocks:
                                if(added_block == block):
                                    added_block.relevant = True
                        else:
                            block.relevant = True
                            required_blocks.append(block)
            # if the class is module level class
            else:
                mro_call_block = context_object.get_mro_function_block('__call__',
                                                                       class_name,
                                                                       program_content)

                for block in all_blocks:
                    # as target blocks are inside some class,
                    # no preprocessing with Block.other_lines is required
                    if(block.start_line >= cls_start
                            and block.end_line <= cls_end
                            and block not in required_blocks):
                        if(block.block_type == CLASS_FUNCTION
                                and block == mro_call_block):
                            block.relevant = True
                        required_blocks.append(block)

                # if `__call__` is inherited
                if(mro_call_block is not None):
                    mro_call_block.relevant = True  # bcs of L129
                    if(mro_call_block not in required_blocks):
                        required_blocks.append(mro_call_block)
        elif(class_type == IMPORT):
            for block in all_blocks:
                # check if the single module_other is
                # also local_block and add to reuired_blocks
                if(block not in required_blocks
                        and block.block_type == MODULE_OTHER):
                    block.relevant = True
                    required_blocks.append(block)

    # update local_block relevance
    # this updation done at the end for correct check of line 121, 147 and 153
    local_block.relevant = True

    return required_blocks
