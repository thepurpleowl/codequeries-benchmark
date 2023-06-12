from get_context import get_span_context
from basecontexts import Block, CLASS_FUNCTION, CLASS_OTHER
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

# train_data: /weblabdeusto/weblabdeusto/server/src/weblab/data/experiments.py
# '''import base64
# import os
# from voodoo.typechecker import typecheck

# class ExperimentId(object):

#     __metaclass__ = Representable

#     @typecheck(basestring, basestring)
#     def __init__(self, exp_name, cat_name):
#         self.exp_name  = unicode(exp_name)
#         self.cat_name  = unicode(cat_name)

#     def __cmp__(self, other):
#         if isinstance(other, ExperimentId):
#             return -1
#         if self.exp_name != other.exp_name:
#             return cmp(self.exp_name, other.exp_name)

#         return cmp(self.cat_name, other.cat_name)

#     def __hash__(self):
#         return hash(self.to_weblab_str())

#     @staticmethod
#     def parse(weblab_str):
#         pos = weblab_str.find("@")
#         experiment_name = weblab_str[:pos]
#         category_name   = weblab_str[pos + 1 :]
#         return ExperimentId(experiment_name, category_name)'''

SOURCE_CODE = ['''import base64\nimport os\nfrom voodoo.typechecker import typecheck\n\nclass ExperimentId(object):\n\n    __metaclass__ = Representable\n\n    @typecheck(basestring, basestring)\n    def __init__(self, exp_name, cat_name):\n        self.exp_name  = unicode(exp_name)\n        self.cat_name  = unicode(cat_name)\n\n    def __cmp__(self, other):\n        if isinstance(other, ExperimentId):\n            return -1\n        if self.exp_name != other.exp_name:\n            return cmp(self.exp_name, other.exp_name)\n\n        return cmp(self.cat_name, other.cat_name)\n\n    def __hash__(self):\n        return hash(self.to_weblab_str())\n\n    @staticmethod\n    def parse(weblab_str):\n        pos = weblab_str.find("@")\n        experiment_name = weblab_str[:pos]\n        category_name   = weblab_str[pos + 1 :]\n        return ExperimentId(experiment_name, category_name)''',
               '''import base64\nimport os\nfrom voodoo.typechecker import typecheck\n\nclass ExperimentId(object):\n\n    __metaclass__ = Representable\n\n    @typecheck(basestring, basestring)\n    def __init__(self, exp_name, cat_name):\n        self.exp_name  = unicode(exp_name)\n        self.cat_name  = unicode(cat_name)\n\n    def __eq__(self, other):\n        return (self.exp_name == other.exp_name)\n\n    def __cmp__(self, other):\n        if isinstance(other, ExperimentId):\n            return -1\n        if self.exp_name != other.exp_name:\n            return cmp(self.exp_name, other.exp_name)\n\n        return cmp(self.cat_name, other.cat_name)\n    @staticmethod\n    def parse(weblab_str):\n        pos = weblab_str.find("@")\n        experiment_name = weblab_str[:pos]\n        category_name   = weblab_str[pos + 1 :]\n        return ExperimentId(experiment_name, category_name)''']
SPAN = [Span(21, 4, 21, 23), Span(13, 4, 13, 28)]
MESSAGE = ['''Class [["ExperimentId"|"relative:///py_file_70710.py:5:1:5:27"]] implements __hash__ but does not define __eq__.''',
           '''Class [["ExperimentId"|"relative:///py_file_70710.py:5:1:5:27"]] implements __eq__ but does not define __hash__.''']


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(8,
                            11,
                            [],
                            '''@typecheck(basestring, basestring)\n    def __init__(self, exp_name, cat_name):\n        self.exp_name  = unicode(exp_name)\n        self.cat_name  = unicode(cat_name)''',
                            'root.ExperimentId.__init__',
                            CLASS_FUNCTION,
                            False,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(13,
                            19,
                            [],
                            '''def __cmp__(self, other):\n        if isinstance(other, ExperimentId):\n            return -1\n        if self.exp_name != other.exp_name:\n            return cmp(self.exp_name, other.exp_name)\n\n        return cmp(self.cat_name, other.cat_name)''',
                            'root.ExperimentId.__cmp__',
                            CLASS_FUNCTION,
                            False,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(21,
                            22,
                            [],
                            '''def __hash__(self):\n        return hash(self.to_weblab_str())''',
                            'root.ExperimentId.__hash__',
                            CLASS_FUNCTION,
                            True,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(24,
                            29,
                            [],
                            '''@staticmethod\n    def parse(weblab_str):\n        pos = weblab_str.find("@")\n        experiment_name = weblab_str[:pos]\n        category_name   = weblab_str[pos + 1 :]\n        return ExperimentId(experiment_name, category_name)''',
                            'root.ExperimentId.parse',
                            CLASS_FUNCTION,
                            False,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(4,
                            29,
                            [4, 5, 6, 7, 12, 20, 23],
                            '''class ExperimentId(object):\n\n    __metaclass__ = Representable\n\n\n\n''',
                            'root.ExperimentId',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object'))],
                     # second list of Blocks
                     [Block(8,
                            11,
                            [],
                            '''@typecheck(basestring, basestring)\n    def __init__(self, exp_name, cat_name):\n        self.exp_name  = unicode(exp_name)\n        self.cat_name  = unicode(cat_name)''',
                            'root.ExperimentId.__init__',
                            CLASS_FUNCTION,
                            False,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(13,
                            14,
                            [],
                            '''def __eq__(self, other):\n        return (self.exp_name == other.exp_name)''',
                            'root.ExperimentId.__eq__',
                            CLASS_FUNCTION,
                            True,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(16,
                            22,
                            [],
                            '''def __cmp__(self, other):\n        if isinstance(other, ExperimentId):\n            return -1\n        if self.exp_name != other.exp_name:\n            return cmp(self.exp_name, other.exp_name)\n\n        return cmp(self.cat_name, other.cat_name)''',
                            'root.ExperimentId.__cmp__',
                            CLASS_FUNCTION,
                            False,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(23,
                            28,
                            [],
                            '''@staticmethod\n    def parse(weblab_str):\n        pos = weblab_str.find("@")\n        experiment_name = weblab_str[:pos]\n        category_name   = weblab_str[pos + 1 :]\n        return ExperimentId(experiment_name, category_name)''',
                            'root.ExperimentId.parse',
                            CLASS_FUNCTION,
                            False,
                            'class ExperimentId(object):',
                            ('__', '__class__', 'ExperimentId')),
                      Block(4,
                            28,
                            [4, 5, 6, 7, 12, 15],
                            '''class ExperimentId(object):\n\n    __metaclass__ = Representable\n\n\n''',
                            'root.ExperimentId',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            msg = MESSAGE[i]
            generated_block = get_span_context('Inconsistent equality and hashing',
                                               code, tree_sitter_parser, '',
                                               msg, span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
