from get_context import get_span_context
from basecontexts import Block, CLASS_FUNCTION, CLASS_OTHER, MODULE_OTHER, STUB
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
    /ployground/ploy/ploy/config.py - builtin-class moule : 7, 7
    /catap/namebench/nb_third_party/dns/rdata.py - class <> : 19, 19
    builtin-class list - 25, 25
'''
# from ploy.common import Hooks

# class HooksMassager():
#     def test_func(self, config, sectionname):
#         hooks = Hooks()
#         for hook_spec in value.split():
#             hooks.add(resolve_dotted_name(hook_spec)())
#         return hooks

# class GenericRdata(Rdata):
#     def __init__(self, rdclass, rdtype, data):
#         super(GenericRdata, self).__init__(rdclass, rdtype)
#         self.data = data

#     def to_wire(self, file, compress = None, origin = None):
#         file.write(self.data)

#     def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin = None):
#         return cls(rdclass, rdtype, wire[current : current + rdlen])

#     from_wire = classmethod(from_wire)

# if __name__ == '__main__':
#     a_list = []
#     a_list()

SOURCE_CODE = ['''from ploy.common import Hooks\n\nclass HooksMassager():\n    def test_func(self, config, sectionname):\n        hooks = Hooks()\n        for hook_spec in value.split():\n            hooks.add(resolve_dotted_name(hook_spec)())\n        return hooks\n\nclass GenericRdata(Rdata):\n    def __init__(self, rdclass, rdtype, data):\n        super(GenericRdata, self).__init__(rdclass, rdtype)\n        self.data = data\n\n    def to_wire(self, file, compress = None, origin = None):\n        file.write(self.data)\n\n    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin = None):\n        return cls(rdclass, rdtype, wire[current : current + rdlen])\n\n    from_wire = classmethod(from_wire)\n\nif __name__ == '__main__':\n    a_list = []\n    a_list()''',
               '''from ploy.common import Hooks\n\nclass HooksMassager():\n    def test_func(self, config, sectionname):\n        hooks = Hooks()\n        for hook_spec in value.split():\n            hooks.add(resolve_dotted_name(hook_spec)())\n        return hooks\n\nclass GenericRdata(Rdata):\n    def __init__(self, rdclass, rdtype, data):\n        super(GenericRdata, self).__init__(rdclass, rdtype)\n        self.data = data\n\n    def to_wire(self, file, compress = None, origin = None):\n        file.write(self.data)\n\n    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin = None):\n        return cls(rdclass, rdtype, wire[current : current + rdlen])\n\n    from_wire = classmethod(from_wire)\n\nif __name__ == '__main__':\n    a_list = []\n    a_list()''',
               '''from ploy.common import Hooks\n\nclass HooksMassager():\n    def test_func(self, config, sectionname):\n        hooks = Hooks()\n        for hook_spec in value.split():\n            hooks.add(resolve_dotted_name(hook_spec)())\n        return hooks\n\nclass GenericRdata(Rdata):\n    def __init__(self, rdclass, rdtype, data):\n        super(GenericRdata, self).__init__(rdclass, rdtype)\n        self.data = data\n\n    def to_wire(self, file, compress = None, origin = None):\n        file.write(self.data)\n\n    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin = None):\n        return cls(rdclass, rdtype, wire[current : current + rdlen])\n\n    from_wire = classmethod(from_wire)\n\nif __name__ == '__main__':\n    a_list = []\n    a_list()''']
MESSAGE = ["builtin-class module:::Call to a [[<>]] of [[<>]].",
           "class GenericRdata:::Call to a [[<>]] of [[<>]].",
           "builtin-class list:::Call to a [[<>]] of [[<>]]."]
SPAN = [Span(6, -1, 6, -1), Span(19, -1, 18, -1), Span(24, -1, 24, -1)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(3,
                            7,
                            [],
                            '''def test_func(self, config, sectionname):\n        hooks = Hooks()\n        for hook_spec in value.split():\n            hooks.add(resolve_dotted_name(hook_spec)())\n        return hooks''',
                            'root.HooksMassager.test_func',
                            CLASS_FUNCTION,
                            True,
                            'class HooksMassager():',
                            ('__', '__class__', 'HooksMassager')),
                     Block(0,
                           24,
                           [0, 1, 8, 21, 22, 23, 24],
                           '''from ploy.common import Hooks\n\n\n\nif __name__ == '__main__':\n    a_list = []\n    a_list()''',
                           'root',
                           MODULE_OTHER,
                           True,
                           'module',
                           ('__', '__class__'))],
                     # second
                     [Block(17,
                            18,
                            [],
                            '''def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin = None):\n        return cls(rdclass, rdtype, wire[current : current + rdlen])''',
                            'root.GenericRdata.from_wire',
                            CLASS_FUNCTION,
                            True,
                            'class GenericRdata(Rdata):',
                            ('__', '__class__', 'GenericRdata', 'Rdata')),
                      Block(10,
                            12,
                            [],
                            '''def __init__(self, rdclass, rdtype, data):\n        super(GenericRdata, self).__init__(rdclass, rdtype)\n        self.data = data''',
                            'root.GenericRdata.__init__',
                            CLASS_FUNCTION,
                            False,
                            'class GenericRdata(Rdata):',
                            ('__', '__class__', 'GenericRdata', 'Rdata')),
                      Block(14,
                            15,
                            [],
                            '''def to_wire(self, file, compress = None, origin = None):\n        file.write(self.data)''',
                            'root.GenericRdata.to_wire',
                            CLASS_FUNCTION,
                            False,
                            'class GenericRdata(Rdata):',
                            ('__', '__class__', 'GenericRdata', 'Rdata')),
                      Block(9,
                            20,
                            [9, 13, 16, 19, 20],
                            '''class GenericRdata(Rdata):\n\n\n\n    from_wire = classmethod(from_wire)''',
                            'root.GenericRdata',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'Rdata'))],
                     # third
                     [Block(0,
                            24,
                            [0, 1, 8, 21, 22, 23, 24],
                            '''from ploy.common import Hooks\n\n\n\nif __name__ == '__main__':\n    a_list = []\n    a_list()''',
                            'root',
                            MODULE_OTHER,
                            True,
                            'module',
                            ('__', '__class__')),
                      Block(-1,
                            -1,
                            [],
                            '''def __call__(self):\n        raise NotImplementedError''',
                            'root.list.__call__',
                            STUB,
                            True,
                            'STUB class list(object):',
                            ('__', '__class__', 'list')),
                      Block(-1,
                            -1,
                            [0],
                            '''class list(object):''',
                            'root.list',
                            STUB,
                            False,
                            'STUB module',
                            ('__', '__class__', 'object'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            message = MESSAGE[i]
            generated_block = get_span_context('Non-callable called',
                                               code, tree_sitter_parser,
                                               '', message, span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
