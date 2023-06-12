import csv
import tempfile
import unittest
import dataset_with_context_pb2
from create_query_result import create_query_result
from create_query_result import create_query_result_merged
from create_tokenized_files_labels import create_tokenized_files_labels


QUERY_DESCRIPTION_TOKENS = [['Implicit', 'string', 'concatenation', 'in', 'a', 'list']]

START_LINE = [[0, 0, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6,
               7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 14, 14,
               14, 14, 15,
               15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19]]


END_LINE = [[0, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8,
             8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 14, 14, 14, 14, 15,
             15, 15, 16, 16,
             16, 17, 17, 17, 18, 18, 18, 19, 19]]

START_COLUMN = [[2, 25, 0, 3, 0, 12, 14, 18, 19, 23, 24, 25, 37,
                 38, 25, 58, 25, 46, 47, 48, 0, 32, 0, 36, 0, 37, 0, 34,
                 0, 25, 0, 17, 0, 0, 8, 10, 11, 40, 42, 11, 33, 34, 11, 42,
                 43, 11, 25, 26, 11, 37, 38, 11]]

END_COLUMN = [[25, 0, 3, 0, 11, 13, 18, 19, 23, 24, 25, 37, 38, 0, 58,
               0, 46, 47, 48, 0, 32, 0, 36, 0, 37, 0, 34, 0, 25, 0, 17, 0, 0,
               7, 9, 11,
               40, 41, 0, 33, 34, 0, 42, 43, 0, 25, 26, 0, 37, 38, 0, 12]]

PROGRAM_TOKEN = [['# -*- coding: utf-8 -*-', '___NL___',
                  '"""\nConnected components.\n"""', '___NEWLINE___',
                  '__authors__', '=',
                  '"\\n"', '.', 'join', '(', '[', "'Eben Kenah'", ',',
                  '___NL___',
                  "'Aric Hagberg (hagberg@lanl.gov)'", '___NL___',
                  "'Christopher Ellison'", ']',
                  ')', '___NEWLINE___', '#    Copyright (C) 2004-2010 by ',
                  '___NL___', '#    Aric Hagberg <hagberg@lanl.gov>',
                  '___NL___',
                  '#    Dan Schult <dschult@colgate.edu>', '___NL___',
                  '#    Pieter Swart <swart@lanl.gov>', '___NL___',
                  '#    All rights reserved.', '___NL___', '#    BSD license.',
                  '___NL___', '___NL___', '__all__', '=', '[',
                  "'number_connected_components'", ',', '___NL___',
                  "'connected_components'", ',', '___NL___',
                  "'connected_component_subgraphs'", ',', '___NL___',
                  "'is_connected'", ',', '___NL___',
                  "'node_connected_component'", ',', '___NL___', ']']]

LABELS = [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
          ]

CSV_FIELDS_RESULTS = ["Name", "Description", "Warning", "Severity",
                      "Path", "Start_line", "Start_column", "End_line",
                      "End_column"]

CSV_ENTRIES_RESULTS = [
    ["Implicit string concatenation in a list",
     "Omitting a comma between strings causes implicit concatenation which \
is confusing in a list.",
     "warning",
     "Implicit string concatenation. Maybe missing a comma?",
     "/gkno/gkno_launcher/src/networkx/algorithms/components/connected.py",
     6, 26, 7, 46]
]


program_dataset = dataset_with_context_pb2.RawProgramDataset()
program = dataset_with_context_pb2.RawProgramFile()
program.file_path.dataset_file_path.source_name = ("eth_py150_open")
program.file_path.dataset_file_path.split = 3
program.file_path.dataset_file_path.unique_file_path = (
    "gkno/gkno_launcher/src/networkx/algorithms/components/connected.py")
program.language = 0
program.file_content = "  # -*- coding: utf-8 -*-\n\"\"\"\nConnected \
components.\n\"\"\"\n__authors__ = \"\\n\".join([\'Eben Kenah\',\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Christopher Ellison\'])\n#    Copyright \
(C) 2004-2010 by \n#    Aric Hagberg <hagberg@lanl.gov>\n#    Dan \
Schult <dschult@colgate.edu>\n#    Pieter Swart <swart@lanl.gov>\n#    \
All rights reserved.\n#    BSD license.\n\n__all__ = \
[\'number_connected_components\', \n           \
\'connected_components\',\n           \'connected_component_subgraphs\',\
\n           \'is_connected\',\n           \'node_connected_component\',\
\n           ]\n"

program_dataset.raw_program_dataset.append(program)


query_dataset = dataset_with_context_pb2.RawQueryList()
query = dataset_with_context_pb2.Query()
query.query_path.repo = "https://github.com/github/codeql"
query.query_path.unique_path = (
    "https://api.github.com/repos/github/codeql/git/blobs/9547d0045cae47e672dc451a0992db3df0c88420"
)
id = bytes("\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8")
query.queryID = id
query.content = "b\'/**\\n * @name Implicit string concatenation in a list\
\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * \
@tags reliability\\n *       maintainability\\n *       convention\\n \
*       external/cwe/cwe-665\\n * @problem.severity warning\\n * \
@sub-severity high\\n * @precision high\\n * @id py/implicit-string-\
concatenation-in-list\\n */\\n\\nimport python\\n\\npredicate \
string_const(Expr s) {\\n  s instanceof StrConst\\n  or\\n  \
string_const(s.(BinaryExpr).getLeft()) and string_const(s.(BinaryExpr).\
getRight())\\n}\\n\\nfrom StrConst s\\nwhere\\n  // Implicitly \
concatenated string is in a list and that list contains at least one \
other string.\\n  exists(List l, Expr other |\\n    not s = other and\
\\n    l.getAnElt() = s and\\n    l.getAnElt() = other and\\n    \
string_const(other)\\n  ) and\\n  exists(s.getAnImplicitlyConcatenatedPart\
()) and\\n  not s.isParenthesized()\\nselect s, \"Implicit string \
concatenation. Maybe missing a comma?\"\\n\'"
query.metadata.name = "Implicit string concatenation in a list"
query.metadata.description = "Omitting a comma between strings causes \
implicit concatenation which is confusing in a list.\\n"
query.metadata.severity = "warning\\n"
query.metadata.message = "NA"
query.metadata.full_metadata = "b\'/**\\n * @name Implicit string concatenation in \
a list\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * @tags \
reliability\\n *       maintainability\\n *       convention\\n *       \
external/cwe/cwe-665\\n * @problem.severity warning\\n * @sub-severity \
high\\n * @precision high\\n * @id py/implicit-string-concatenation-in-\
list\\n */\\n\\n"
query.language = 0
query.hops = 1
query.span = 2
query_dataset.raw_query_set.append(query)


def generate_csv_file(CSV_FIELDS, CSV_ENTRIES):
    csv_file_obj = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv")
    filename = csv_file_obj.name

    # create a csv writer object and write
    csvwriter = csv.writer(csv_file_obj)
    csvwriter.writerow(CSV_FIELDS)
    csvwriter.writerows(CSV_ENTRIES)
    csv_file_obj.flush()

    return filename, csv_file_obj


def generate_csv_file_no_column_names(CSV_ENTRIES):
    csv_file_obj = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv")
    filename = csv_file_obj.name

    # create a csv writer object and write
    csvwriter = csv.writer(csv_file_obj)
    csvwriter.writerows(CSV_ENTRIES)
    csv_file_obj.flush()

    return filename, csv_file_obj


class TestCreateTokenizedFilesLabels(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    dataset = create_query_result(program_dataset, query_dataset,
                                  results_file, 'positive')

    dataset_merged = create_query_result_merged(dataset)

    tokenized_dataset = create_tokenized_files_labels(dataset_merged, 'positive')

    def test_query_description_tokens(self):
        for i in range(len(self.tokenized_dataset.tokens_and_labels)):
            self.assertEqual(
                self.tokenized_dataset.tokens_and_labels[i].
                query_name_tokens, QUERY_DESCRIPTION_TOKENS[i])

    def test_tokens_metadata_labels(self):
        for i in range(len(self.tokenized_dataset.tokens_and_labels)):
            for j in range(len(self.tokenized_dataset.tokens_and_labels[i].
                               tokens_metadata_labels)):
                self.assertEqual(
                    self.tokenized_dataset.tokens_and_labels[i].
                    tokens_metadata_labels[j].start_line, START_LINE[i][j])

                self.assertEqual(
                    self.tokenized_dataset.tokens_and_labels[i].
                    tokens_metadata_labels[j].end_line, END_LINE[i][j])

                self.assertEqual(
                    self.tokenized_dataset.tokens_and_labels[i].
                    tokens_metadata_labels[j].start_column, START_COLUMN[i][j])

                self.assertEqual(
                    self.tokenized_dataset.tokens_and_labels[i].
                    tokens_metadata_labels[j].end_column, END_COLUMN[i][j])

                self.assertEqual(
                    self.tokenized_dataset.tokens_and_labels[i].
                    tokens_metadata_labels[j].program_token,
                    PROGRAM_TOKEN[i][j])

                self.assertEqual(
                    self.tokenized_dataset.tokens_and_labels[i].tokens_metadata_labels[j].label,
                    LABELS[i][j]
                )

    def test_query_and_files_results(self):
        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_path.dataset_file_path.source_name,
            "eth_py150_open"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_path.dataset_file_path.split, 3
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_path.dataset_file_path.
            unique_file_path, "gkno/gkno_launcher/src/networkx/\
algorithms/components/connected.py"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.language, 0
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_content,
            "  # -*- coding: utf-8 -*-\n\"\"\"\nConnected \
components.\n\"\"\"\n__authors__ = \"\\n\".join([\'Eben Kenah\',\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Christopher Ellison\'])\n#    Copyright \
(C) 2004-2010 by \n#    Aric Hagberg <hagberg@lanl.gov>\n#    Dan \
Schult <dschult@colgate.edu>\n#    Pieter Swart <swart@lanl.gov>\n#    \
All rights reserved.\n#    BSD license.\n\n__all__ = \
[\'number_connected_components\', \n           \
\'connected_components\',\n           \'connected_component_subgraphs\',\
\n           \'is_connected\',\n           \'node_connected_component\',\
\n           ]\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].start_line,
            6
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].end_line,
            7
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].start_column,
            26
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].end_column,
            46
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.query_path.repo,
            "https://github.com/github/codeql"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.query_path.unique_path,
            "https://api.github.com/repos/github/codeql/git/blobs/\
9547d0045cae47e672dc451a0992db3df0c88420"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.query.
            queryID,
            bytes(
                "\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8"
            )
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.content,
            "b\'/**\\n * @name Implicit string concatenation in a list\
\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * \
@tags reliability\\n *       maintainability\\n *       convention\\n \
*       external/cwe/cwe-665\\n * @problem.severity warning\\n * \
@sub-severity high\\n * @precision high\\n * @id py/implicit-string-\
concatenation-in-list\\n */\\n\\nimport python\\n\\npredicate \
string_const(Expr s) {\\n  s instanceof StrConst\\n  or\\n  \
string_const(s.(BinaryExpr).getLeft()) and string_const(s.(BinaryExpr).\
getRight())\\n}\\n\\nfrom StrConst s\\nwhere\\n  // Implicitly \
concatenated string is in a list and that list contains at least one \
other string.\\n  exists(List l, Expr other |\\n    not s = other and\
\\n    l.getAnElt() = s and\\n    l.getAnElt() = other and\\n    \
string_const(other)\\n  ) and\\n  exists(s.getAnImplicitlyConcatenatedPart\
()) and\\n  not s.isParenthesized()\\nselect s, \"Implicit string \
concatenation. Maybe missing a comma?\"\\n\'"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.name,
            "Implicit string concatenation in a list"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.description,
            "Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.severity,
            "warning\\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.message,
            "NA"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.full_metadata,
            "b\'/**\\n * @name Implicit string concatenation in \
a list\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * @tags \
reliability\\n *       maintainability\\n *       convention\\n *       \
external/cwe/cwe-665\\n * @problem.severity warning\\n * @sub-severity \
high\\n * @precision high\\n * @id py/implicit-string-concatenation-in-\
list\\n */\\n\\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.language,
            0
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.hops,
            1
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.span,
            2
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_path.dataset_file_path.source_name,
            "eth_py150_open"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_path.dataset_file_path.split, 3
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            raw_file_path.file_path.dataset_file_path.
            unique_file_path, "gkno/gkno_launcher/src/networkx/algorithms/components/connected.py"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.raw_file_path.
            language, 0
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.raw_file_path.
            file_content,
            "  # -*- coding: utf-8 -*-\n\"\"\"\nConnected \
components.\n\"\"\"\n__authors__ = \"\\n\".join([\'Eben Kenah\',\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Christopher Ellison\'])\n#    Copyright \
(C) 2004-2010 by \n#    Aric Hagberg <hagberg@lanl.gov>\n#    Dan \
Schult <dschult@colgate.edu>\n#    Pieter Swart <swart@lanl.gov>\n#    \
All rights reserved.\n#    BSD license.\n\n__all__ = \
[\'number_connected_components\', \n           \
\'connected_components\',\n           \'connected_component_subgraphs\',\
\n           \'is_connected\',\n           \'node_connected_component\',\
\n           ]\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].start_line,
            6
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].end_line,
            7
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].start_column,
            26
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            resultlocation[0].end_column,
            46
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.query_path.repo,
            "https://github.com/github/codeql"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.query_path.unique_path,
            "https://api.github.com/repos/github/codeql/git/blobs/9547d0045cae47e672dc451a0992db3df0c88420"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.query.
            queryID,
            bytes("\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8")
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.content,
            "b\'/**\\n * @name Implicit string concatenation in a list\
\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * \
@tags reliability\\n *       maintainability\\n *       convention\\n \
*       external/cwe/cwe-665\\n * @problem.severity warning\\n * \
@sub-severity high\\n * @precision high\\n * @id py/implicit-string-\
concatenation-in-list\\n */\\n\\nimport python\\n\\npredicate \
string_const(Expr s) {\\n  s instanceof StrConst\\n  or\\n  \
string_const(s.(BinaryExpr).getLeft()) and string_const(s.(BinaryExpr).\
getRight())\\n}\\n\\nfrom StrConst s\\nwhere\\n  // Implicitly \
concatenated string is in a list and that list contains at least one \
other string.\\n  exists(List l, Expr other |\\n    not s = other and\
\\n    l.getAnElt() = s and\\n    l.getAnElt() = other and\\n    \
string_const(other)\\n  ) and\\n  exists(s.getAnImplicitlyConcatenatedPart\
()) and\\n  not s.isParenthesized()\\nselect s, \"Implicit string \
concatenation. Maybe missing a comma?\"\\n\'"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.name,
            "Implicit string concatenation in a list"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.description,
            "Omitting a comma between strings causes \
implicit concatenation which is confusing in a list.\\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.severity,
            "warning\\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.message,
            "NA"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.metadata.full_metadata,
            "b\'/**\\n * @name Implicit string concatenation in \
a list\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * @tags \
reliability\\n *       maintainability\\n *       convention\\n *       \
external/cwe/cwe-665\\n * @problem.severity warning\\n * @sub-severity \
high\\n * @precision high\\n * @id py/implicit-string-concatenation-in-\
list\\n */\\n\\n"
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.language,
            0
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.hops,
            1
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.span,
            2
        )

        self.assertEqual(
            self.tokenized_dataset.tokens_and_labels[0].
            query_and_files_results.
            query.distributable,
            False
        )


if __name__ == "__main__":
    unittest.main()
