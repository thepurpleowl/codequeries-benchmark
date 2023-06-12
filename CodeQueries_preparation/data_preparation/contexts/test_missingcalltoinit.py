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

# class Vehicle(object):

#     def __init__(self):
#         self.mobile = True

# class Car(Vehicle):

#     def __init__(self):
#         Vehicle.__init__(self)
#         self.car_init()

# #Car.__init__ is missed out.
# class SportsCar(Car, Vehicle):

#     def __init__(self):
#         Vehicle.__init__(self)
#         self.sports_car_init()

# #Fix SportsCar by calling Car.__init__
# class FixedSportsCar(Car, Vehicle):

#     def __init__(self):
#         Car.__init__(self)
#         self.sports_car_init()

SOURCE_CODE = ['''class Vehicle(object):\n    \n    def __init__(self):\n        self.mobile = True\n        \nclass Car(Vehicle):\n    \n    def __init__(self):\n        Vehicle.__init__(self)\n        self.car_init()\n        \n#Car.__init__ is missed out.\nclass SportsCar(Car, Vehicle):\n    \n    def __init__(self):\n        Vehicle.__init__(self)\n        self.sports_car_init()\n        \n#Fix SportsCar by calling Car.__init__\nclass FixedSportsCar(Car, Vehicle):\n    \n    def __init__(self):\n        Car.__init__(self)\n        self.sports_car_init()''']
SPAN = [Span(12, 0, 12, 29)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(2,
                            3,
                            [],
                            'def __init__(self):\n        self.mobile = True',
                            'root.Vehicle.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Vehicle(object):',
                            ('__', '__class__', 'Vehicle')),
                      Block(7,
                            9,
                            [],
                            'def __init__(self):\n        Vehicle.__init__(self)\n        self.car_init()',
                            'root.Car.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class Car(Vehicle):',
                            ('__', '__class__', 'Car', 'Vehicle')),
                      Block(14,
                            16,
                            [],
                            'def __init__(self):\n        Vehicle.__init__(self)\n        self.sports_car_init()',
                            'root.SportsCar.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class SportsCar(Car, Vehicle):',
                            ('__', '__class__', 'SportsCar', 'Car', 'Vehicle')),
                      Block(12,
                            16,
                            [12, 13],
                            'class SportsCar(Car, Vehicle):\n    ',
                            'root.SportsCar',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__', 'Car', 'Vehicle'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('Missing call to `__init__` during object initialization',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
