from get_context import get_span_context
from basecontexts import Block, CLASS_OTHER
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

# train_data: mutated- /peterhudec/authomatic/tests/functional_tests/expected_values/tumblr.py
# SOURCE_CODE = [
# '''class Person:
#     "This is a person class"
#     age = 10

#     def greet(self):
#         print('Hello')

#     birth_year = 2000
#     current_year = birth_year + age

# print(Person.age)

# def get_age(age):
#     print (age)

# # create a new object of Person class
# harry = Person()
# # Calling object's greet() method
# harry.greet()''']

SOURCE_CODE = ['''class Person:\n    "This is a person class"\n    age = 10\n\n    def greet(self):\n        print('Hello')\n\n    birth_year = 2000\n    current_year = birth_year + age\n\nprint(Person.age)\n\ndef get_age(age):\n    print (age)\n\n# create a new object of Person class\nharry = Person()\n# Calling object's greet() method\nharry.greet()''']
SPAN = [Span(8, -1, 8, -1)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(0,
                            8,
                            [0, 1, 2, 3, 6, 7, 8],
                            '''class Person:\n    "This is a person class"\n    age = 10\n\n\n    birth_year = 2000\n    current_year = birth_year + age''',
                            'root.Person',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            generated_block = get_span_context('Redundant assignment',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[j][i], gen_block)


if __name__ == "__main__":
    unittest.main()
