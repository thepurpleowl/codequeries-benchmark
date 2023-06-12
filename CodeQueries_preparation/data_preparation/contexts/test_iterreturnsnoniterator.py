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

# from gluon.tools import Crud

# class MongoCursorWrapper:
#     def __init__ (self, cursor):
#         self.__cursor = cursor

#     def __iter__ (self):
#         return MongoWrapperIter (self.__cursor)

# class MongoWrapper:
#     def __init__ (self, cursor):
#         self.__dict__['cursor'] = cursor

#     def __nonzero__ (self):
#         if self.cursor is None:
#             return False
#         return len (self.cursor) != 0

#     def __iter__ (self):
#         return MongoWrapperIter (self.cursor)

# class MongoWrapperIter:
#     def __init__ (self, cursor):
#         self.__cursor = iter (cursor)

#     def __iter__ (self):
#         return self

SOURCE_CODE = ['''from gluon.tools import Crud\n\nclass MongoCursorWrapper:\n    def __init__ (self, cursor):\n        self.__cursor = cursor\n    \n    def __iter__ (self):\n        return MongoWrapperIter (self.__cursor)\n\nclass MongoWrapper:\n    def __init__ (self, cursor):\n        self.__dict__['cursor'] = cursor\n\n    def __nonzero__ (self):\n        if self.cursor is None:\n            return False\n        return len (self.cursor) != 0\n\n    def __iter__ (self):\n        return MongoWrapperIter (self.cursor)\n\nclass MongoWrapperIter:\n    def __init__ (self, cursor):\n        self.__cursor = iter (cursor)\n\n    def __iter__ (self):\n        return self''']
MESSAGE = ['''Class MongoWrapperIter is returned as an iterator (by [["__iter__"|"relative:///py_file_466.py:7:5:7:24"]]) but does not fully implement the iterator interface.\nClass MongoWrapperIter is returned as an iterator (by [["__iter__"|"relative:///py_file_466.py:19:5:19:24"]]) but does not fully implement the iterator interface.''']
SPAN = [Span(21, 0, 21, 23)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(6,
                            7,
                            [],
                            '''def __iter__ (self):\n        return MongoWrapperIter (self.__cursor)''',
                            'root.MongoCursorWrapper.__iter__',
                            CLASS_FUNCTION,
                            True,
                            'class MongoCursorWrapper:',
                            ('__', '__class__', 'MongoCursorWrapper')),
                      Block(18,
                            19,
                            [],
                            '''def __iter__ (self):\n        return MongoWrapperIter (self.cursor)''',
                            'root.MongoWrapper.__iter__',
                            CLASS_FUNCTION,
                            True,
                            'class MongoWrapper:',
                            ('__', '__class__', 'MongoWrapper')),
                      Block(25,
                            26,
                            [],
                            '''def __iter__ (self):\n        return self''',
                            'root.MongoWrapperIter.__iter__',
                            CLASS_FUNCTION,
                            True,
                            'class MongoWrapperIter:',
                            ('__', '__class__', 'MongoWrapperIter')),
                      Block(21,
                            26,
                            [21, 24],
                            '''class MongoWrapperIter:\n''',
                            'root.MongoWrapperIter',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            message = MESSAGE[i]
            generated_block = get_span_context('`__iter__` method returns a non-iterator',
                                               code, tree_sitter_parser, '',
                                               message, span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
