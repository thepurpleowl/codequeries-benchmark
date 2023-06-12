from get_context import get_span_context
from basecontexts import Block, CLASS_FUNCTION, CLASS_OTHER
from tree_sitter import Language, Parser
from collections import namedtuple
import unittest
# to handle import while stand-alone test
import sys
sys.path.insert(0, '..')
PY_LANGUAGE = Language("../my-languages.so", "python")

# # to handle bazel tests
# PATH_PREFIX = "../code-cubert/data_preparation/"
# PY_LANGUAGE = Language(PATH_PREFIX + "my-languages.so", "python")

tree_sitter_parser = Parser()
tree_sitter_parser.set_language(PY_LANGUAGE)

Span = namedtuple('Span', 'start_line start_col end_line end_col')

# class Super(object):
#     def __init__(self, arg):
#         self._state = "Not OK"
#         self.set_up(arg)
#         self._state = "OK"

#     def set_up(self, arg):
#         "Do some set up"

# class Sub(Super):
#     def __init__(self, arg):
#         Super.__init__(self, arg)
#         self.important_state = "OK"

#     def set_up(self, arg):
#         Super.set_up(self, arg)

SOURCE_CODE = ['''class Super(object):\n    def __init__(self, arg):\n        self._state = "Not OK"\n        self.set_up(arg)\n        self._state = "OK"\n\n    def set_up(self, arg):\n        "Do some set up"\n\nclass Sub(Super):\n    def __init__(self, arg):\n        Super.__init__(self, arg)\n        self.important_state = "OK"\n\n    def set_up(self, arg):\n        Super.set_up(self, arg)''']
SPAN = [Span(3, 8, 3, 24)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(10,
                            12,
                            [],
                            '''def __init__(self, arg):\n        Super.__init__(self, arg)\n        self.important_state = "OK"''',
                            'root.Sub.__init__',
                            CLASS_FUNCTION,
                            False,
                            'class Sub(Super):',
                            ('__', '__class__', 'Sub', 'Super')),
                      Block(14,
                            15,
                            [],
                            '''def set_up(self, arg):\n        Super.set_up(self, arg)''',
                            'root.Sub.set_up',
                            CLASS_FUNCTION,
                            True,
                            'class Sub(Super):',
                            ('__', '__class__', 'Sub', 'Super')),
                      Block(9,
                            15,
                            [9, 13],
                            '''class Sub(Super):\n''',
                            'root.Sub',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'Super')),
                      Block(1,
                            4,
                            [],
                            '''def __init__(self, arg):\n        self._state = "Not OK"\n        self.set_up(arg)\n        self._state = "OK"''',
                            'root.Super.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Super(object):',
                            ('__', '__class__', 'Super')),
                      Block(6,
                            7,
                            [],
                            '''def set_up(self, arg):\n        "Do some set up"''',
                            'root.Super.set_up',
                            CLASS_FUNCTION,
                            False,
                            'class Super(object):',
                            ('__', '__class__', 'Super')),
                      Block(0,
                            7,
                            [0, 5],
                            '''class Super(object):\n''',
                            'root.Super',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('`__init__` method calls overridden method',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
