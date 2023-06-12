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

# class Point(object):

#     def __init__(self, x, y):
#         self._x = x
#         self._y = y

#     def __repr__(self):
#         return 'Point(%r, %r)' % (self._x, self._y)

#     def __eq__(self, other):
#         if not isinstance(other, Point):
#             return False
#         return self._x == other._x and self._y == other._y

# class ColorPoint(Point):

#     def __init__(self, x, y, color):
#         Point.__init__(self, x, y)
#         self._color = color

#     def __repr__(self):
#         return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)

SOURCE_CODE = ['''class Point(object):\n\n    def __init__(self, x, y):\n        self._x = x\n        self._y = y\n\n    def __repr__(self):\n        return 'Point(%r, %r)' % (self._x, self._y)\n\n    def __eq__(self, other):\n        if not isinstance(other, Point):\n            return False\n        return self._x == other._x and self._y == other._y\n\nclass ColorPoint(Point):\n\n    def __init__(self, x, y, color):\n        Point.__init__(self, x, y)\n        self._color = color\n\n    def __repr__(self):\n        return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)''',
               '''class Point(object):\n\n    def __init__(self, x, y):\n        self._x = x\n        self._y = y\n\n    def __repr__(self):\n        return 'Point(%r, %r)' % (self._x, self._y)\n\n    def __eq__(self, other):\n        if not isinstance(other, Point):\n            return False\n        return self._x == other._x and self._y == other._y\n\nclass ColorPoint(Point, Color):\n\n    def __init__(self, x, y, color):\n        Color.__init__(self, x, y)\n        self._color = color\n\n    def __repr__(self):\n        return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)\n\nclass Color(object):\n\n    def __init__(self, color):\n        self.color = color\n\n    def __eq__(self, other):\n        return self.color == other.color\n''',
               '''class Point(object):\n\n    def __init__(self, x, y):\n        self._x = x\n        self._y = y\n\n    def __repr__(self):\n        return 'Point(%r, %r)' % (self._x, self._y)\n\n    def __eq__(self, other):\n        if not isinstance(other, Point):\n            return False\n        return self._x == other._x and self._y == other._y\n\nclass ColorPoint(Point, Color):\n\n    def __init__(self, x, y, color):\n        super(ColorPoint, self).__init__(self, x, y)\n        self._color = color\n\n    def __repr__(self):\n        return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)\n\nclass Color(object):\n\n    def __init__(self, color):\n        self.color = color\n\n    def __eq__(self, other):\n        return self.color == other.color\n''']
SPAN = [Span(14, 0, 14, 24), Span(14, 0, 14, 31), Span(14, 0, 14, 31)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(2,
                            4,
                            [],
                            '''def __init__(self, x, y):\n        self._x = x\n        self._y = y''',
                            'root.Point.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(6,
                            7,
                            [],
                            '''def __repr__(self):\n        return 'Point(%r, %r)' % (self._x, self._y)''',
                            'root.Point.__repr__',
                            CLASS_FUNCTION,
                            False,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(9,
                            12,
                            [],
                            '''def __eq__(self, other):\n        if not isinstance(other, Point):\n            return False\n        return self._x == other._x and self._y == other._y''',
                            'root.Point.__eq__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(16,
                            18,
                            [],
                            '''def __init__(self, x, y, color):\n        Point.__init__(self, x, y)\n        self._color = color''',
                            'root.ColorPoint.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class ColorPoint(Point):',
                            ('__', '__class__', 'ColorPoint', 'Point')),
                      Block(20,
                            21,
                            [],
                            '''def __repr__(self):\n        return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)''',
                            'root.ColorPoint.__repr__',
                            CLASS_FUNCTION,
                            False,
                            'class ColorPoint(Point):',
                            ('__', '__class__', 'ColorPoint', 'Point')),
                      Block(0,
                            12,
                            [0, 1, 5, 8],
                            '''class Point(object):\n\n\n''',
                            'root.Point',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object')),
                      Block(14,
                            21,
                            [14, 15, 19],
                            '''class ColorPoint(Point):\n\n''',
                            'root.ColorPoint',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__', 'Point'))],
                     # second
                     [Block(2,
                            4,
                            [],
                            '''def __init__(self, x, y):\n        self._x = x\n        self._y = y''',
                            'root.Point.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(6,
                            7,
                            [],
                            '''def __repr__(self):\n        return 'Point(%r, %r)' % (self._x, self._y)''',
                            'root.Point.__repr__',
                            CLASS_FUNCTION,
                            False,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(9,
                            12,
                            [],
                            '''def __eq__(self, other):\n        if not isinstance(other, Point):\n            return False\n        return self._x == other._x and self._y == other._y''',
                            'root.Point.__eq__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(16,
                            18,
                            [],
                            '''def __init__(self, x, y, color):\n        Color.__init__(self, x, y)\n        self._color = color''',
                            'root.ColorPoint.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class ColorPoint(Point, Color):',
                            ('__', '__class__', 'ColorPoint', 'Point', 'Color')),
                      Block(20,
                            21,
                            [],
                            '''def __repr__(self):\n        return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)''',
                            'root.ColorPoint.__repr__',
                            CLASS_FUNCTION,
                            False,
                            'class ColorPoint(Point, Color):',
                            ('__', '__class__', 'ColorPoint', 'Point', 'Color')),
                      Block(25,
                            26,
                            [],
                            '''def __init__(self, color):\n        self.color = color''',
                            'root.Color.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Color(object):',
                            ('__', '__class__', 'Color')),
                      Block(28,
                            29,
                            [],
                            '''def __eq__(self, other):\n        return self.color == other.color''',
                            'root.Color.__eq__',
                            CLASS_FUNCTION,
                            False,
                            'class Color(object):',
                            ('__', '__class__', 'Color')),
                      Block(0,
                            12,
                            [0, 1, 5, 8],
                            '''class Point(object):\n\n\n''',
                            'root.Point',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object')),
                      Block(14,
                            21,
                            [14, 15, 19],
                            '''class ColorPoint(Point, Color):\n\n''',
                            'root.ColorPoint',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__', 'Point', 'Color')),
                      Block(23,
                            29,
                            [23, 24, 27],
                            'class Color(object):\n\n',
                            'root.Color',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object'))],
                     # third
                     [Block(2,
                            4,
                            [],
                            '''def __init__(self, x, y):\n        self._x = x\n        self._y = y''',
                            'root.Point.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(6,
                            7,
                            [],
                            '''def __repr__(self):\n        return 'Point(%r, %r)' % (self._x, self._y)''',
                            'root.Point.__repr__',
                            CLASS_FUNCTION,
                            False,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(9,
                            12,
                            [],
                            '''def __eq__(self, other):\n        if not isinstance(other, Point):\n            return False\n        return self._x == other._x and self._y == other._y''',
                            'root.Point.__eq__',
                            CLASS_FUNCTION,
                            True,
                            'class Point(object):',
                            ('__', '__class__', 'Point')),
                      Block(16,
                            18,
                            [],
                            '''def __init__(self, x, y, color):\n        super(ColorPoint, self).__init__(self, x, y)\n        self._color = color''',
                            'root.ColorPoint.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class ColorPoint(Point, Color):',
                            ('__', '__class__', 'ColorPoint', 'Point', 'Color')),
                      Block(20,
                            21,
                            [],
                            '''def __repr__(self):\n        return 'ColorPoint(%r, %r)' % (self._x, self._y, self._color)''',
                            'root.ColorPoint.__repr__',
                            CLASS_FUNCTION,
                            False,
                            'class ColorPoint(Point, Color):',
                            ('__', '__class__', 'ColorPoint', 'Point', 'Color')),
                      Block(25,
                            26,
                            [],
                            '''def __init__(self, color):\n        self.color = color''',
                            'root.Color.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Color(object):',
                            ('__', '__class__', 'Color')),
                      Block(28,
                            29,
                            [],
                            '''def __eq__(self, other):\n        return self.color == other.color''',
                            'root.Color.__eq__',
                            CLASS_FUNCTION,
                            False,
                            'class Color(object):',
                            ('__', '__class__', 'Color')),
                      Block(0,
                            12,
                            [0, 1, 5, 8],
                            '''class Point(object):\n\n\n''',
                            'root.Point',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object')),
                      Block(14,
                            21,
                            [14, 15, 19],
                            '''class ColorPoint(Point, Color):\n\n''',
                            'root.ColorPoint',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__', 'Point', 'Color')),
                      Block(23,
                            29,
                            [23, 24, 27],
                            'class Color(object):\n\n',
                            'root.Color',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('`__eq__` not overridden when adding attributes',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
