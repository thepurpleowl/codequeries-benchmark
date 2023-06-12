from get_context import get_span_context
from basecontexts import Block, CLASS_FUNCTION
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

# import logging
# log = logging.getLogger(__name__)

# class Resource():
#     def get(self, request):
#         self.instance = request
#         return self.instance

# class NonFieldResource(Resource):
#     def __init__(self):
#         self.rsc = None

# class FieldResource():
#     def get(self, request, pk):
#         instance = self.get_object(request, pk=pk)
#         return self.prepare(request, instance)

# class FieldsResource(FieldResource, NonFieldResource):
#     def is_not_found(self, request, response, *args, **kwargs):
#         return False

#     def get(self, request):
#         params = self.get_params(request)
#         queryset = self.get_queryset(request, params)

SOURCE_CODE = ['''import logging\nlog = logging.getLogger(__name__)\n\nclass Resource():\n    def get(self, request):\n        self.instance = request\n        return self.instance\n\nclass NonFieldResource(Resource):\n    def __init__(self):\n        self.rsc = None\n\nclass FieldResource():\n    def get(self, request, pk):\n        instance = self.get_object(request, pk=pk)\n        return self.prepare(request, instance)\n\nclass FieldsResource(FieldResource, NonFieldResource):\n    def is_not_found(self, request, response, *args, **kwargs):\n        return False\n\n    def get(self, request):\n        params = self.get_params(request)\n        queryset = self.get_queryset(request, params)''']
SPAN = [Span(21, 4, 21, 27)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(18,
                            19,
                            [],
                            '''def is_not_found(self, request, response, *args, **kwargs):\n        return False''',
                            'root.FieldsResource.is_not_found',
                            CLASS_FUNCTION,
                            False,
                            'class FieldsResource(FieldResource, NonFieldResource):',
                            ('__', '__class__', 'FieldsResource', 'FieldResource', 'NonFieldResource', 'Resource')),
                      Block(21,
                            23,
                            [],
                            'def get(self, request):\n        params = self.get_params(request)\n        queryset = self.get_queryset(request, params)',
                            'root.FieldsResource.get',
                            'CLASS_FUNCTION',
                            True,
                            'class FieldsResource(FieldResource, NonFieldResource):',
                            ('__', '__class__', 'FieldsResource', 'FieldResource', 'NonFieldResource', 'Resource')),
                      Block(17,
                            23,
                            [17, 20],
                            'class FieldsResource(FieldResource, NonFieldResource):\n',
                            'root.FieldsResource',
                            'CLASS_OTHER',
                            False,
                            'module',
                            ('__', '__class__', 'FieldResource', 'NonFieldResource')),
                      Block(4,
                            6,
                            [],
                            'def get(self, request):\n        self.instance = request\n        return self.instance',
                            'root.Resource.get',
                            'CLASS_FUNCTION',
                            False,
                            'class Resource():',
                            ('__', '__class__', 'Resource')),
                      Block(9,
                            10,
                            [],
                            'def __init__(self):\n        self.rsc = None',
                            'root.NonFieldResource.__init__',
                            'CLASS_FUNCTION',
                            False,
                            'class NonFieldResource(Resource):',
                            ('__', '__class__', 'NonFieldResource', 'Resource')),
                      Block(13,
                            15,
                            [],
                            'def get(self, request, pk):\n        instance = self.get_object(request, pk=pk)\n        return self.prepare(request, instance)',
                            'root.FieldResource.get',
                            'CLASS_FUNCTION',
                            True,
                            'class FieldResource():',
                            ('__', '__class__', 'FieldResource')),
                      Block(3,
                            6,
                            [3],
                            'class Resource():',
                            'root.Resource',
                            'CLASS_OTHER',
                            False,
                            'module',
                            ('__', '__class__')),
                      Block(8,
                            10,
                            [8],
                            'class NonFieldResource(Resource):',
                            'root.NonFieldResource',
                            'CLASS_OTHER',
                            False,
                            'module',
                            ('__', '__class__', 'Resource')),
                      Block(12,
                            15,
                            [12],
                            'class FieldResource():',
                            'root.FieldResource',
                            'CLASS_OTHER',
                            False,
                            'module',
                            ('__', '__class__'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('Signature mismatch in overriding method',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
