from get_context import get_span_context
from basecontexts import Block, MODULE_FUNCTION, CLASS_FUNCTION, CLASS_OTHER, STUB
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

'''
    builtin-class list - 5, 5
    /nigelsmall/httpstream/httpstream/tardis.py - 28, 28
'''
# from datetime import timedelta

# default = []
# def checkempty(border=default):
#     if border is default:
#         return None

# class timezone(tzinfo):
#     _Omitted = object()

#     def __new__(cls, offset, name=_Omitted):
#         if not isinstance(offset, timedelta):
#             raise TypeError("offset must be a timedelta")
#         if name is cls._Omitted:
#             if not offset:
#                 return cls.utc
#             name = None
#         elif not isinstance(name, str):
#             raise TypeError("name must be a string")
#         return cls._create(offset, name)

#     def __eq__(self, other):
#         if type(other) != timezone:
#             return False
#         return self._offset == other._offset

#     def __repr__(self):
#         if self is self.utc:
#             return datetime.timezone.utc

SOURCE_CODE = ['''from datetime import timedelta\n\ndefault = []\ndef checkempty(border=default):\n    if border is default:\n        return None\n\nclass timezone(tzinfo):\n    _Omitted = object()\n\n    def __new__(cls, offset, name=_Omitted):\n        if not isinstance(offset, timedelta):\n            raise TypeError("offset must be a timedelta")\n        if name is cls._Omitted:\n            if not offset:\n                return cls.utc\n            name = None\n        elif not isinstance(name, str):\n            raise TypeError("name must be a string")\n        return cls._create(offset, name)\n\n    def __eq__(self, other):\n        if type(other) != timezone:\n            return False\n        return self._offset == other._offset\n\n    def __repr__(self):\n        if self is self.utc:\n            return datetime.timezone.utc''',
               '''from datetime import timedelta\n\ndefault = []\ndef checkempty(border=default):\n    if border is default:\n        return None\n\nclass timezone(tzinfo):\n    _Omitted = object()\n\n    def __new__(cls, offset, name=_Omitted):\n        if not isinstance(offset, timedelta):\n            raise TypeError("offset must be a timedelta")\n        if name is cls._Omitted:\n            if not offset:\n                return cls.utc\n            name = None\n        elif not isinstance(name, str):\n            raise TypeError("name must be a string")\n        return cls._create(offset, name)\n\n    def __eq__(self, other):\n        if type(other) != timezone:\n            return False\n        return self._offset == other._offset\n\n    def __repr__(self):\n        if self is self.utc:\n            return datetime.timezone.utc''']
MESSAGE = ["builtin-class list:::Values compared using 'is' when equivalence is not the same as identity. Use '==' instead.",
           "class timezone:::Values compared using 'is' when equivalence is not the same as identity. Use '==' instead."]
SPAN = [Span(4, 7, 4, 25), Span(27, 11, 27, 28)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(3,
                            5,
                            [],
                            '''def checkempty(border=default):\n    if border is default:\n        return None''',
                            'root.checkempty',
                            MODULE_FUNCTION,
                            True,
                            'module',
                            ('__', '__class__')),
                     Block(-1,
                           -1,
                           [],
                           '''def __eq__(self, value):\n        return self==value''',
                           'root.list.__eq__',
                           STUB,
                           True,
                           'class list(object):',
                           ('__', '__class__')),
                     Block(-1,
                           -1,
                           [0],
                           '''class list():''',
                           'root.list',
                           STUB,
                           False,
                           'class list(object):',
                           ('__', '__class__'))],
                     # second
                     [Block(26,
                            28,
                            [],
                            '''def __repr__(self):\n        if self is self.utc:\n            return datetime.timezone.utc''',
                            'root.timezone.__repr__',
                            CLASS_FUNCTION,
                            True,
                            'class timezone(tzinfo):',
                            ('__', '__class__', 'timezone', 'tzinfo')),
                      Block(10,
                            19,
                            [],
                            '''def __new__(cls, offset, name=_Omitted):\n        if not isinstance(offset, timedelta):\n            raise TypeError("offset must be a timedelta")\n        if name is cls._Omitted:\n            if not offset:\n                return cls.utc\n            name = None\n        elif not isinstance(name, str):\n            raise TypeError("name must be a string")\n        return cls._create(offset, name)''',
                            'root.timezone.__new__',
                            CLASS_FUNCTION,
                            False,
                            'class timezone(tzinfo):',
                            ('__', '__class__', 'timezone', 'tzinfo')),
                      Block(21,
                            24,
                            [],
                            '''def __eq__(self, other):\n        if type(other) != timezone:\n            return False\n        return self._offset == other._offset''',
                            'root.timezone.__eq__',
                            CLASS_FUNCTION,
                            True,
                            'class timezone(tzinfo):',
                            ('__', '__class__', 'timezone', 'tzinfo')),
                      Block(7,
                            28,
                            [7, 8, 9, 20, 25],
                            '''class timezone(tzinfo):\n    _Omitted = object()\n\n\n''',
                            'root.timezone',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'tzinfo'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            message = MESSAGE[i]
            generated_block = get_span_context('Comparison using is when operands support `__eq__`',
                                               code, tree_sitter_parser,
                                               '', message, span, None)

        for j, gen_block in enumerate(generated_block):
            self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
