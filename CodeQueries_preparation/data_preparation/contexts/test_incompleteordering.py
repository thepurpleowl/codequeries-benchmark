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

tree_sitter_parser = Parser()
tree_sitter_parser.set_language(PY_LANGUAGE)

Span = namedtuple('Span', 'start_line start_col end_line end_col')

# train_data: /clips/pattern/pattern/server/cherrypy/cherrypy/lib/httputil.py
# import re
# import urllib


# def protocol_from_http(protocol_str):
#     return int(protocol_str[5]), int(protocol_str[7])

# class HeaderElement(object):
#     def __init__(self, value, params=None):
#         self.value = value
#         if params is None:
#             params = {}
#         self.params = params

#     def __cmp__(self, other):
#         return cmp(self.value, other.value)

#     def __lt__(self, other):
#         return self.value < other.value

#     def parse(elementstr):
#         atoms = [x.strip() for x in elementstr.split(";") if x.strip()]
#         if not atoms:
#             initial_value = ''
#         else:
#             initial_value = atoms.pop(0).strip()
#         return initial_value, params
#     parse = staticmethod(parse)

SOURCE_CODE = ['''import os\n\ncurr_dir = os.cwd()\n\nclass Element(object):\n    """\n        Base Element class\n    """\n    def __init__(self, value):\n        self.value = value\n\n    def __le__(self, other):\n        return self.value <= other.value\n\nclass HeaderElement(HHElement):\n    def __init__(self, value, params=None):\n        super().__init__(value)\n        self.value = value\n        if params is None:\n            params = {}\n        self.params = params\n\n    def __cmp__(self, other):\n        return cmp(self.value, other.value)\n\n    def __lt__(self, other):\n        return self.value < other.value\n\n    def parse(elementstr):\n        atoms = [x.strip() for x in elementstr.split(";") if x.strip()]\n        if not atoms:\n            initial_value = \'\'\n        else:\n            initial_value = atoms.pop(0).strip()\n        return initial_value, params\n    parse = staticmethod(parse)\n\nclass HHElement(Element):\n    """\n        Base Element class\n    """\n    def __init__(self, value):\n        self.value = value\n\n    def __ge__(self, other):\n        return self.value <= other.value''']
SPAN = [Span(14, 0, 14, 31)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(15,
                            20,
                            [],
                            '''def __init__(self, value, params=None):\n        super().__init__(value)\n        self.value = value\n        if params is None:\n            params = {}\n        self.params = params''',
                            'root.HeaderElement.__init__',
                            CLASS_FUNCTION,
                            False,
                            'class HeaderElement(HHElement):',
                            ('__', '__class__', 'HeaderElement', 'HHElement', 'Element')),
                      Block(22,
                            23,
                            [],
                            '''def __cmp__(self, other):\n        return cmp(self.value, other.value)''',
                            'root.HeaderElement.__cmp__',
                            CLASS_FUNCTION,
                            False,
                            'class HeaderElement(HHElement):',
                            ('__', '__class__', 'HeaderElement', 'HHElement', 'Element')),
                      Block(25,
                            26,
                            [],
                            '''def __lt__(self, other):\n        return self.value < other.value''',
                            'root.HeaderElement.__lt__',
                            CLASS_FUNCTION,
                            True,
                            'class HeaderElement(HHElement):',
                            ('__', '__class__', 'HeaderElement', 'HHElement', 'Element')),
                      Block(28,
                            34,
                            [],
                            '''def parse(elementstr):\n        atoms = [x.strip() for x in elementstr.split(";") if x.strip()]\n        if not atoms:\n            initial_value = \'\'\n        else:\n            initial_value = atoms.pop(0).strip()\n        return initial_value, params''',
                            'root.HeaderElement.parse',
                            CLASS_FUNCTION,
                            False,
                            'class HeaderElement(HHElement):',
                            ('__', '__class__', 'HeaderElement', 'HHElement', 'Element')),
                      Block(14,
                            35,
                            [14, 21, 24, 27, 35],
                            '''class HeaderElement(HHElement):\n\n\n\n    parse = staticmethod(parse)''',
                            'root.HeaderElement',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__', 'HHElement')),
                      Block(44,
                            45,
                            [],
                            '''def __ge__(self, other):\n        return self.value <= other.value''',
                            'root.HHElement.__ge__',
                            CLASS_FUNCTION,
                            True,
                            'class HHElement(Element):',
                            ('__', '__class__', 'HHElement', 'Element')),
                      Block(11,
                            12,
                            [],
                            '''def __le__(self, other):\n        return self.value <= other.value''',
                            'root.Element.__le__',
                            CLASS_FUNCTION,
                            True,
                            'class Element(object):',
                            ('__', '__class__', 'Element'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('Incomplete ordering',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
