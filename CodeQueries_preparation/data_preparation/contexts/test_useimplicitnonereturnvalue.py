from get_context import get_span_context, __columns__
from basecontexts import Block, MODULE_FUNCTION, MODULE_OTHER
from tree_sitter import Language, Parser
from collections import namedtuple
import pandas as pd
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

# SOURCE_CODE = [
# '''import os
# import numpy as np
#
# class Employee:
#     def __init__(self, id, name):
#         self.id = id
#         self.name = name
#
# class SalaryEmployee(Employee):
#     def __init__(self, id, name, weekly_salary):
#         super().__init__(id, name)
#         self.weekly_salary = weekly_salary
#         self.total = self.calculate_payroll()
#
#     def calculate_payroll(self):
#         return self.weekly_salary
#
# class HourlyEmployee(Employee):
#     def __init__(self, id, name, hours_worked, hour_rate):
#         super().__init__(id, name)
#         self.hours_worked = hours_worked
#         self.hour_rate = hour_rate
#         self.total = self.calculate_payroll()
#
#     def calculate_payroll(self):
#         return self.hours_worked * self.hour_rate
#
# class FulltimeEmployee(Employee):
#     def __init__(self, id, name, employee_type='F'):
#         super().__init__(id, name)
#         if(employee_type=='F'):
#             self.ratio = np.array([0.5, 0.3, 0.2])
#         else:
#             self.ratio = np.array([1, 0, 0])
#
#     def work(self):
#         print(f'{self.name} gets {self.div} in compensation')
#
# class Intern(HourlyEmployee, FulltimeEmployee):
#     def __init__(self, id, name, hours_worked, hour_rate):
#         super(Intern, self).__init__(id, name, hours_worked, hour_rate)
#         self.bonus = 0
#
#     def get_bonus(self, hours):
#         if(self.hours_worked < 10):
#             self.bonus = 2
#         else:
#             self.bonus = 10
#
#     def get_work_desc(self, name, hours, bonus):
#         return self.name + " worked " + str(hours) + " hours | " + str(self.div) + " : added bonus - " + str(bonus)
#
#     def work(self):
#         self.div = np.array(self.ratio * np.array([self.total for i in range(len(self.ratio))]), dtype=int)
#         self.bonus = self.get_bonus(self.hours_worked)
#         if(self.bonus > 0):
#             self.bonus = self.bonus + get_festive_bonus()
#         desc = self.get_work_desc(self.name, self.bonus)
#         print(desc)
#
# def get_festive_bonus(intern_obj):
#     intern_obj.bonus = intern_obj.bonus * 1.2
#
# intern = Intern(4, "II1", 20, 1.5)
# intern.work()
# print(get_festive_bonus(intern))''']

SOURCE_CODE = ['''import os\nimport numpy as np\n\nclass Employee:\n    def __init__(self, id, name):\n        self.id = id\n        self.name = name\n\nclass SalaryEmployee(Employee):\n    def __init__(self, id, name, weekly_salary):\n        super().__init__(id, name)\n        self.weekly_salary = weekly_salary\n        self.total = self.calculate_payroll()\n\n    def calculate_payroll(self):\n        return self.weekly_salary\n\t\nclass HourlyEmployee(Employee):\n    def __init__(self, id, name, hours_worked, hour_rate):\n        super().__init__(id, name)\n        self.hours_worked = hours_worked\n        self.hour_rate = hour_rate\n        self.total = self.calculate_payroll()\n\n    def calculate_payroll(self):\n        return self.hours_worked * self.hour_rate\n\nclass FulltimeEmployee(Employee):\n    def __init__(self, id, name, employee_type=\'F\'):\n        super().__init__(id, name)\n        if(employee_type==\'F\'):\n            self.ratio = np.array([0.5, 0.3, 0.2])\n        else:\n            self.ratio = np.array([1, 0, 0])\n\n    def work(self):\n        print(f\'{self.name} gets {self.div} in compensation\')\n\nclass Intern(HourlyEmployee, FulltimeEmployee):\n    def __init__(self, id, name, hours_worked, hour_rate):\n        super(Intern, self).__init__(id, name, hours_worked, hour_rate)\n        self.bonus = 0\n\n    def get_bonus(self, hours):\n        if(self.hours_worked < 10):\n            self.bonus = 2\n        else:\n            self.bonus = 10\n    \n    def get_work_desc(self, name, hours, bonus):\n        return self.name + " worked " + str(hours) + " hours | " + str(self.div) + " : added bonus - " + str(bonus)\n    \n    def work(self):\n        self.div = np.array(self.ratio * np.array([self.total for i in range(len(self.ratio))]), dtype=int)\n        self.bonus = self.get_bonus(self.hours_worked)\n        if(self.bonus > 0):\n            self.bonus = self.bonus + get_festive_bonus()\n        desc = self.get_work_desc(self.name, self.bonus)\n        print(desc)\n\ndef get_festive_bonus(intern_obj):\n    intern_obj.bonus = intern_obj.bonus * 1.2\n\nintern = Intern(4, "II1", 20, 1.5)\nintern.work()\nprint(get_festive_bonus(intern))''']
SPANS = [Span(65, 6, 65, 31)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(0,
                            65,
                            [0, 1, 2, 7, 16, 26, 37, 59, 62, 63, 64, 65],
                            '''import os\nimport numpy as np\n\n\n\t\n\n\n\n\nintern = Intern(4, "II1", 20, 1.5)\nintern.work()\nprint(get_festive_bonus(intern))''',
                            'root',
                            MODULE_OTHER,
                            True,
                            'module',
                            ('__', '__class__')),
                     Block(60,
                           61,
                           [],
                           '''def get_festive_bonus(intern_obj):\n    intern_obj.bonus = intern_obj.bonus * 1.2''',
                           'root.get_festive_bonus',
                           MODULE_FUNCTION,
                           True,
                           'module',
                           ('__', '__class__'))]]

    def test_relevant_block(self):
        test_aux_result_df = pd.read_csv('./test_data/test__aux_res.csv',
                                         names=__columns__)
        for i, code in enumerate(SOURCE_CODE):
            span = SPANS[i]
            generated_block = get_span_context('Use of the return value of a procedure',
                                               code, tree_sitter_parser, 'test_unused_imp.py',
                                               '', span, test_aux_result_df)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
