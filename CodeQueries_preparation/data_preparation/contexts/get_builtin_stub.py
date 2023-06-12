# from basecontexts import Block, ROOT_BLOCK_TYPE, STUB


call_stub = '''\
class <builtin_type>(object):
    def __call__(self):
        raise NotImplementedError'''

eq_stub = '''\
class <builtin_type>(object):
    def __eq__(self, value):
        return self==value'''

custom_function_stub = '''\
class <builtin_type>(object):
    def <function_name>(self):
        raise NotImplementedError'''


def get_class_specific_call_stub(builtin_type):
    """
    This function returns specifc builtin class stub.
    Args:
        builtin_type: builtin object type in python
    Returns:
        A Block with stub __call__ content
    """
    class_specific_stub = call_stub.replace('<builtin_type>', builtin_type)
    # stub_block = Block(-1, -1, [], class_specific_stub, ROOT_BLOCK_TYPE, STUB)

    return class_specific_stub


def get_class_specific_eq_stub(builtin_type):
    """
    This function returns specifc builtin class stub.
    Args:
        builtin_type: builtin object type in python
    Returns:
        A Block with stub __eq__ content
    """
    class_specific_stub = eq_stub.replace('<builtin_type>', builtin_type)
    # stub_block = Block(-1, -1, [], class_specific_stub, ROOT_BLOCK_TYPE, STUB)

    return class_specific_stub


def get_custom_stub(builtin_type, function_name):
    """
    This function returns specifc builtin class stub.
    Args:
        builtin_type: builtin object type in python
        function_name: required function name for STUB block
    Returns:
        A CLASS_FUNCTION Block with function_name and builtin_type
    """
    custom_stub = custom_function_stub.replace('<builtin_type>', builtin_type)
    custom_stub = custom_stub.replace('<function_name>', function_name)
    # stub_block = Block(-1, -1, [], class_specific_stub, ROOT_BLOCK_TYPE, STUB)

    return custom_stub
