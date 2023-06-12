from get_context import get_span_context
from basecontexts import Block, MODULE_FUNCTION, CLASS_FUNCTION, CLASS_OTHER
from tree_sitter import Language, Parser
from collections import namedtuple
import unittest
# to handle import while stand-alone test
import sys
sys.path.insert(0, '..')
PY_LANGUAGE = Language("../my-languages.so", "python")

# to handle bazel tests
# PATH_PREFIX = "../code-cubert/data_preparation/"
# PY_LANGUAGE = Language(PATH_PREFIX + "my-languages.so", "python")

tree_sitter_parser = Parser()
tree_sitter_parser.set_language(PY_LANGUAGE)

Span = namedtuple('Span', 'start_line start_col end_line end_col')

# class Point(object):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def sum(self, x, y):
#         return x + y

# def get_obj():
#     p = Point(1,2,3)

# if __name__ == '__main__':
#     get_obj()

SOURCE_CODE = ['''class Point(object):\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def sum(self, x, y):\n        return x + y\n\ndef get_obj():\n    p = Point(1,2,3)\n\nif __name__ == '__main__':\n    get_obj()''']
SPAN = [Span(9, 8, 9, 20)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(8,
                            9,
                            [],
                            '''def get_obj():\n    p = Point(1,2,3)''',
                            'root.get_obj',
                            MODULE_FUNCTION,
                            True,
                            'module',
                            ('__', '__class__')),
                      Block(1,
                            3,
                            [],
                            '''def __init__(self, x, y):\n        self.x = x\n        self.y = y''',
                            'root.Point.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                     Block(5,
                           6,
                           [],
                           '''def sum(self, x, y):\n        return x + y''',
                           'root.Point.sum',
                           CLASS_FUNCTION,
                           False,
                           'class Point(object):',
                           ('__', '__class__', 'Point')),
                     Block(0,
                           6,
                           [0, 4],
                           '''class Point(object):\n''',
                           'root.Point',
                           CLASS_OTHER,
                           False,
                           'module',
                           ('__', '__class__', 'object'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('Wrong number of arguments in a class instantiation',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
