import unittest
import csv
import tempfile
import dataset_with_context_pb2
from create_query_result import create_query_result, create_query_result_merged


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
     6, 26, 7, 58],
    ["Implicit string concatenation in a list",
     "Omitting a comma between strings causes implicit concatenation which \
is confusing in a list.",
     "warning",
     "Implicit string concatenation. Maybe missing a comma?",
     "/gkno/gkno_launcher/src/networkx/algorithms/components/connected.py",
     7, 26, 8, 46],
    ["Backspace escape in regular expression",
     "Using '\b' to escape the backspace character in a regular expression is \
confusing since it could be mistaken for a word boundary assertion.",
     "recommendation",
     "Backspace escape in regular expression at offset 13.",
     "/AppScale/appscale/AppServer/google/appengine/_internal/django/utils/\
simplejson/encoder.py", 8, 21, 8, 47],
    ["`__iter__` method returns a non-iterator",
     """The `__iter__` method returns a non-iterator which, if used in a 'for' loop, would raise a 'TypeError'""",
     "error",
     """Class CommentStripper is returned as an iterator \
      (by [["__iter__"|"relative:///py_file_29137.py:79:5:79:23"]]) but does not fully implement the iterator interface.""",
     "/CGATOxford/cgat/CGAT/CSV.py", 69, 1, 69, 22],
    ["Unreachable code",
     "Code is unreachable",
     "warning",
     "Unreachable statement.",
     "/CGATOxford/cgat/CGAT/CSV.py", 221, 17, 221, 32],
    ["`__iter__` method returns a non-iterator",
     """The `__iter__` method returns a non-iterator which, if used in a 'for' loop, would raise a 'TypeError'""",
     "error",
     """Class DictReaderLarge is returned as an iterator (by [["__iter__"|"relative:///py_file_29137.py:148:5:148:23"]]) \
        but does not fully implement the iterator interface.""",
     "/CGATOxford/cgat/CGAT/CSV.py", 132, 1, 132, 22],
    ["`__iter__` method returns a non-iterator",
     """The `__iter__` method returns a non-iterator which, if used in a 'for' loop, would raise a 'TypeError'""",
     "error",
     """Class UnicodeCsvReader is returned as an iterator (by [["__iter__"|"relative:///py_file_29137.py:111:5:111:23"]]) \
     but does not fully implement the iterator interface.""",
     "/CGATOxford/cgat/CGAT/CSV.py", 105, 1, 105, 31]
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

program = dataset_with_context_pb2.RawProgramFile()
program.file_path.dataset_file_path.source_name = (
    "eth_py150_open"
)
program.file_path.dataset_file_path.split = 3
program.file_path.dataset_file_path.unique_file_path = (
    "AppScale/appscale/AppServer/google/appengine/_internal/django/utils/\
simplejson/encoder.py"
)
program.language = 0
program.file_content = "\"\"\"Implementation of JSONEncoder\n\"\"\"\
\nimport re\n\nc_encode_basestring_ascii = None\nc_make_encoder = \
None\n\nESCAPE = re.compile(r\'[\\x00 -\\x1f\\\\\"\\b\\f\\n\\r\\t]\'\
)\nESCAPE_ASCII = re.compile(r\'([\\\\\"]|[ ^\\ -~])\')\nHAS_UTF8 = \
re.compile(r\'[\\x80 -\\xff]\')\nESCAPE_DCT = {\n    \'\\\\\': \'\
\\\\\\\\\', \n    \'\"\': \'\\\\\"\', \n    \'\\b\': \'\\\\b\', \
\n    \'\\f\': \'\\\\f\', \n    \'\\n\': \'\\\\n\', \n    \'\\r\': \
\'\\\\r\', \n    \'\\t\': \'\\\\t\', \n}\n"

program_dataset.raw_program_dataset.append(program)

program = dataset_with_context_pb2.RawProgramFile()
program.file_path.dataset_file_path.source_name = (
    "eth_py150_open"
)
program.file_path.dataset_file_path.split = 3
program.file_path.dataset_file_path.unique_file_path = ("CGATOxford/cgat/CGAT/CSV.py")
program.language = 0
with open('./contexts/test_data/test_file_content.py', 'r') as f:
    program.file_content = f.read()

program_dataset.raw_program_dataset.append(program)

query_dataset = dataset_with_context_pb2.RawQueryList()
query = dataset_with_context_pb2.Query()
query.query_path.repo = "https://github.com/github/codeql"
query.query_path.unique_path = (
    "https://api.github.com/repos/github/codeql/git/blobs/\
9547d0045cae47e672dc451a0992db3df0c88420"
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

query = dataset_with_context_pb2.Query()
query.query_path.repo = "https://github.com/github/codeql"
query.query_path.unique_path = "https://api.github.com/repos/github/codeql/\
git/blobs/ce69dabec44bc874e9780ff7e0b1992b669518e0"
id = bytes("-2\016\325p\350_,\245g\032\237\262\274\343\207", "utf-8")
query.queryID = id

query.content = "b\'/**\\n * @name Backspace escape in regular expression\
\\n * @description Using \\\'\\\\b\\\' to escape the backspace character \
in a regular expression is confusing\\n *              since it could be \
mistaken for a word boundary assertion.\\n * @kind problem\\n * @tags \
maintainability\\n * @problem.severity recommendation\\n * \
@sub-severity high\\n * @precision very-high\\n * @id py/regex\
/backspace-escape\\n */\\n\\nimport python\\nimport semmle.python.\
regex\\n\\nfrom Regex r, int offset\\nwhere\\n  r.escapingChar(offset) \
and\\n  r.getChar(offset + 1) = \"b\" and\\n  exists(int start, int end | \
start < offset and end > offset | r.charSet(start, end))\\nselect r, \"\
Backspace escape in regular expression at offset \" + offset + \".\"\\n\'"
query.metadata.name = "Backspace escape in regular expression"
query.metadata.description = "Using \\\'\\\\b\\\' to escape the backspace \
character in a regular expression is confusing\\n               \
since it could be mistaken for a word boundary assertion.\\n"

query.metadata.severity = "recommendation\\n"
query.metadata.message = "NA"
query.metadata.full_metadata = "b\'/**\\n * @name Backspace escape in regular \
expression\\n * @description Using \\\'\\\\b\\\' to escape the \
backspace character in a regular expression is confusing\\n \
*              since it could be mistaken for a word boundary \
assertion.\\n * @kind problem\\n * @tags maintainability\\n * \
@problem.severity recommendation\\n * @sub-severity high\\n * @precision \
very-high\\n * @id py/regex/backspace-escape\\n */\\n\\n"
query.language = 0
query.hops = 1
query.span = 2
query_dataset.raw_query_set.append(query)

query = dataset_with_context_pb2.Query()
query.query_path.repo = "https://github.com/github/codeql"
query.query_path.unique_path = (
    "https://api.github.com/repos/github/codeql/git/blobs/ed4a240ec4d1066dfe008d4e19a66ef6bb187b46"
)
id = bytes("\223FK\311MECUZ:^\000I\213\256\"", "utf-8")
query.queryID = id
query.content = "/**\n * @name `__iter__` method returns a non-iterator\n * \
@description The `__iter__` method returns a non-iterator which, if used in a \'for\' loop, would raise a \'TypeError\'.\n * \
@kind problem\n * @tags reliability\n *       correctness\n * @problem.severity error\n * @sub-severity low\n * \
@precision high\n * @id py/iter-returns-non-iterator\n */\n\nimport python\n\n\
from ClassValue iterable, FunctionValue iter, ClassValue iterator\nwhere\n  iter = iterable.lookup(\"__iter__\") \
and\n  iterator = iter.getAnInferredReturnType() and\n  not iterator.isIterator()\nselect iterator,\
\n  \"Class \" + iterator.getName() +\n    \" is returned as an iterator (by $@) but does not fully \
implement the iterator interface.\",\n  iter, iter.getName()\n"
query.metadata.name = "`__iter__` method returns a non-iterator"
query.metadata.description = "The `__iter__` method returns a non-iterator which, if used in a \'for\' loop, would raise a \'TypeError\'."
query.metadata.severity = "error\\n"
query.metadata.message = "NA"
query.metadata.full_metadata = "b\'/**\n * @name `__iter__` method returns a non-iterator\n * \
@description The `__iter__` method returns a non-iterator which, if used in a \'for\' loop, would raise a \'TypeError\'.\n * \
@kind problem\n * @tags reliability\n *       correctness\n * @problem.severity error\n * @sub-severity low\n * \
@precision high\n * @id py/iter-returns-non-iterator\n */\n\n"
query.language = 0
query.distributable = False
query.span = 1
query_dataset.raw_query_set.append(query)

query = dataset_with_context_pb2.Query()
query.query_path.repo = "https://github.com/github/codeql"
query.query_path.unique_path = (
    "https://api.github.com/repos/github/codeql/git/blobs/04e9f79c415f2ca2f2dcf17b8e1ff18dbf395425"
)
id = bytes("\335\ni\327\241\363~(\312\302\304\177~\375B\017", "utf-8")
query.queryID = id
query.content = "/**\n * @name Unreachable code\n * @description Code is unreachable\n * \
@kind problem\n * @tags maintainability\n *       useless-code\n *       external/cwe/cwe-561\n * @problem.severity warning\
\n * @sub-severity low\n * @precision very-high\n * @id py/unreachable-statement\n */\n\nimport python\n\n\
predicate typing_import(ImportingStmt is) {\n  exists(Module m |\n    is.getScope() = m and\n    exists(TypeHintComment \
tc | tc.getLocation().getFile() = m.getFile())\n  )\n}\n\n/** Holds if `s` contains the only `yield` in scope */\n\
predicate unique_yield(Stmt s) {\n  exists(Yield y | s.contains(y)) and\n  exists(Function f |\n    f = s.getScope() \
and\n    strictcount(Yield y | f.containsInScope(y)) = 1\n  )\n}\n\n/** Holds if `contextlib.suppress` may be used \
in the same scope as `s` */\npredicate suppression_in_scope(Stmt s) {\n  exists(With w |\n    \
w.getContextExpr().(Call).getFunc().pointsTo(Value::named(\"contextlib.suppress\")) and\n    w.getScope() = s.getScope()\n  \
)\n}\n\n/** Holds if `s` is a statement that raises an exception at the end of an if-elif-else chain. */\npredicate \
marks_an_impossible_else_branch(Stmt s) {\n  exists(If i | i.getOrelse().getItem(0) = s |\n    s.(Assert).getTest() instanceof \
False\n    or\n    s instanceof Raise\n  )\n}\n\npredicate reportable_unreachable(Stmt s) {\n  s.isUnreachable() \
and\n  not typing_import(s) and\n  not suppression_in_scope(s) and\n  not exists(Stmt other | other.isUnreachable() |\n    \
other.contains(s)\n    or\n    exists(StmtList l, int i, int j | l.getItem(i) = other and l.getItem(j) = s and i < j)\n  ) and\n  \
not unique_yield(s) and\n  not marks_an_impossible_else_branch(s)\n}\n\n\
from Stmt s\nwhere reportable_unreachable(s)\nselect s, \"Unreachable statement.\"\n"
query.metadata.name = "Unreachable code"
query.metadata.description = "Code is unreachable"
query.metadata.severity = "warning\\n"
query.metadata.message = "NA"
query.metadata.full_metadata = "/**\n * @name Unreachable code\n * @description Code is unreachable\n * \
@kind problem\n * @tags maintainability\n *       useless-code\n *       external/cwe/cwe-561\n * @problem.severity warning\
\n * @sub-severity low\n * @precision very-high\n * @id py/unreachable-statement\n */\n\n"
query.language = 0
query.distributable = True
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


class TestCreateQueryResult(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    dataset = create_query_result(program_dataset, query_dataset,
                                  results_file, 'positive')

    def test_raw_file(self):
        source = self.dataset.query_and_files_results[0].raw_file.\
            file_path.dataset_file_path.source_name
        self.assertEqual(source, "eth_py150_open")

        split = self.dataset.query_and_files_results[0].raw_file.\
            file_path.dataset_file_path.split
        self.assertEqual(split, 3)

        path = self.dataset.query_and_files_results[0].raw_file.\
            file_path.dataset_file_path.unique_file_path
        self.assertEqual(
            path,
            "gkno/gkno_launcher/src/networkx/algorithms/components/\
connected.py"
        )

        language = self.dataset.query_and_files_results[0].raw_file.\
            language
        self.assertEqual(language, 0)

        content = self.dataset.query_and_files_results[0].raw_file.file_content
        actual_content = "  # -*- coding: utf-8 -*-\n\"\"\"\nConnected \
components.\n\"\"\"\n__authors__ = \"\\n\".join([\'Eben Kenah\',\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Christopher Ellison\'])\n#    Copyright (C) \
2004-2010 by \n#    Aric Hagberg <hagberg@lanl.gov>\n#    Dan Schult \
<dschult@colgate.edu>\n#    Pieter Swart <swart@lanl.gov>\n#    All rights \
reserved.\n#    BSD license.\n\n__all__ = [\'number_connected_components\', \
\n           \'connected_components\',\n           \'\
connected_component_subgraphs\',\n           \'is_connected\',\
\n           \'node_connected_component\',\n           ]\n"
        self.assertEqual(content, actual_content)

    def test_raw_query(self):
        repo = self.dataset.query_and_files_results[0].query.query_path.repo
        self.assertEqual(repo, "https://github.com/github/codeql")

        path = self.dataset.query_and_files_results[0].query.query_path.\
            unique_path
        self.assertEqual(
            path,
            "https://api.github.com/repos/github/codeql/git/blobs/\
9547d0045cae47e672dc451a0992db3df0c88420"
        )

        actual_id = bytes(
            "\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8"
        )
        id = self.dataset.query_and_files_results[0].query.queryID
        self.assertEqual(id, actual_id)

        actual_content = "b\'/**\\n * @name Implicit string concatenation in a \
list\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * @tags \
reliability\\n *       maintainability\\n *       convention\\n *       \
external/cwe/cwe-665\\n * @problem.severity warning\\n * @sub-severity high\
\\n * @precision high\\n * @id py/implicit-string-concatenation-in-list\\n \
*/\\n\\nimport python\\n\\npredicate string_const(Expr s) {\\n  s instanceof \
StrConst\\n  or\\n  string_const(s.(BinaryExpr).getLeft()) and string_const(s.\
(BinaryExpr).getRight())\\n}\\n\\nfrom StrConst s\\nwhere\\n  // Implicitly \
concatenated string is in a list and that list contains at least one other \
string.\\n  exists(List l, Expr other |\\n    not s = other and\\n    l.\
getAnElt() = s and\\n    l.getAnElt() = other and\\n    string_const(other)\
\\n  ) and\\n  exists(s.getAnImplicitlyConcatenatedPart()) and\\n  not s.\
isParenthesized()\\nselect s, \"Implicit string concatenation. Maybe missing \
a comma?\"\\n\'"

        content = self.dataset.query_and_files_results[0].query.content
        self.assertEqual(actual_content, content)

        actual_description = "Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n"
        description = self.dataset.query_and_files_results[0].\
            query.metadata.description
        self.assertEqual(actual_description, description)

        actual_severity = "warning\\n"
        severity = self.dataset.query_and_files_results[0].\
            query.metadata.severity
        self.assertEqual(actual_severity, severity)

        actual_message = "NA"
        message = self.dataset.query_and_files_results[0].\
            query.metadata.message
        self.assertEqual(actual_message, message)

        language = self.dataset.query_and_files_results[0].\
            query.language
        self.assertEqual(language, 0)

        hop = self.dataset.query_and_files_results[0].\
            query.hops
        self.assertEqual(1, hop)

        span = self.dataset.query_and_files_results[0].\
            query.span
        self.assertEqual(2, span)

    def test_result_location(self):
        start_line = self.dataset.query_and_files_results[0].\
            result_location.start_line
        self.assertEqual(start_line, 6)

        end_line = self.dataset.query_and_files_results[0].\
            result_location.end_line
        self.assertEqual(end_line, 7)

        start_column = self.dataset.query_and_files_results[0].\
            result_location.start_column
        self.assertEqual(start_column, 26)

        end_column = self.dataset.query_and_files_results[0].\
            result_location.end_column
        self.assertEqual(end_column, 58)


class TestCreateMergedQueryResult(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    dataset = create_query_result(program_dataset, query_dataset,
                                  results_file, 'positive')
    merged_query_dataset = create_query_result_merged(dataset)

    def test_raw_file(self):
        source = self.merged_query_dataset.query_and_files_results[0].raw_file_path.\
            file_path.dataset_file_path.source_name
        self.assertEqual(source, "eth_py150_open")

        split = self.merged_query_dataset.query_and_files_results[0].raw_file_path.\
            file_path.dataset_file_path.split
        self.assertEqual(split, 3)

        path = self.merged_query_dataset.query_and_files_results[0].raw_file_path.\
            file_path.dataset_file_path.unique_file_path
        self.assertEqual(
            path,
            "gkno/gkno_launcher/src/networkx/algorithms/components/connected.py"
        )

        language = self.merged_query_dataset.query_and_files_results[0].raw_file_path.\
            language
        self.assertEqual(language, 0)

        content = self.merged_query_dataset.query_and_files_results[0].raw_file_path.file_content
        actual_content = "  # -*- coding: utf-8 -*-\n\"\"\"\nConnected \
components.\n\"\"\"\n__authors__ = \"\\n\".join([\'Eben Kenah\',\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Aric Hagberg (hagberg@lanl.gov)\'\
\n                         \'Christopher Ellison\'])\n#    Copyright (C) \
2004-2010 by \n#    Aric Hagberg <hagberg@lanl.gov>\n#    Dan Schult \
<dschult@colgate.edu>\n#    Pieter Swart <swart@lanl.gov>\n#    All rights \
reserved.\n#    BSD license.\n\n__all__ = [\'number_connected_components\', \
\n           \'connected_components\',\n           \'\
connected_component_subgraphs\',\n           \'is_connected\',\
\n           \'node_connected_component\',\n           ]\n"
        self.assertEqual(content, actual_content)

    def test_raw_query(self):
        repo = self.merged_query_dataset.query_and_files_results[0].query.query_path.repo
        self.assertEqual(repo, "https://github.com/github/codeql")

        path = self.merged_query_dataset.query_and_files_results[0].query.query_path.\
            unique_path
        self.assertEqual(
            path,
            "https://api.github.com/repos/github/codeql/git/blobs/\
9547d0045cae47e672dc451a0992db3df0c88420"
        )

        actual_id = bytes(
            "\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8"
        )
        id = self.merged_query_dataset.query_and_files_results[0].query.queryID
        self.assertEqual(id, actual_id)

        actual_content = "b\'/**\\n * @name Implicit string concatenation in a \
list\\n * @description Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n * @kind problem\\n * @tags \
reliability\\n *       maintainability\\n *       convention\\n *       \
external/cwe/cwe-665\\n * @problem.severity warning\\n * @sub-severity high\
\\n * @precision high\\n * @id py/implicit-string-concatenation-in-list\\n \
*/\\n\\nimport python\\n\\npredicate string_const(Expr s) {\\n  s instanceof \
StrConst\\n  or\\n  string_const(s.(BinaryExpr).getLeft()) and string_const(s.\
(BinaryExpr).getRight())\\n}\\n\\nfrom StrConst s\\nwhere\\n  // Implicitly \
concatenated string is in a list and that list contains at least one other \
string.\\n  exists(List l, Expr other |\\n    not s = other and\\n    l.\
getAnElt() = s and\\n    l.getAnElt() = other and\\n    string_const(other)\
\\n  ) and\\n  exists(s.getAnImplicitlyConcatenatedPart()) and\\n  not s.\
isParenthesized()\\nselect s, \"Implicit string concatenation. Maybe missing \
a comma?\"\\n\'"

        content = self.merged_query_dataset.query_and_files_results[0].query.content
        self.assertEqual(actual_content, content)

        actual_description = "Omitting a comma between strings causes implicit \
concatenation which is confusing in a list.\\n"
        description = self.merged_query_dataset.query_and_files_results[0].\
            query.metadata.description
        self.assertEqual(actual_description, description)

        actual_severity = "warning\\n"
        severity = self.merged_query_dataset.query_and_files_results[0].\
            query.metadata.severity
        self.assertEqual(actual_severity, severity)

        actual_message = "NA"
        message = self.merged_query_dataset.query_and_files_results[0].\
            query.metadata.message
        self.assertEqual(actual_message, message)

        language = self.merged_query_dataset.query_and_files_results[0].\
            query.language
        self.assertEqual(language, 0)

        hop = self.merged_query_dataset.query_and_files_results[0].\
            query.hops
        self.assertEqual(1, hop)

        span = self.merged_query_dataset.query_and_files_results[0].\
            query.span
        self.assertEqual(2, span)

        actual_full_metadata = "/**\n * @name Unreachable code\n * @description Code is unreachable\n * \
@kind problem\n * @tags maintainability\n *       useless-code\n *       external/cwe/cwe-561\n * @problem.severity warning\
\n * @sub-severity low\n * @precision very-high\n * @id py/unreachable-statement\n */\n\n"
        full_metadata = self.merged_query_dataset.query_and_files_results[3].query.metadata.full_metadata
        self.assertEqual(actual_full_metadata, full_metadata)

    def test_result_location(self):
        # first result location
        start_line = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[0].start_line
        self.assertEqual(start_line, 6)

        end_line = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[0].end_line
        self.assertEqual(end_line, 7)

        start_column = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[0].start_column
        self.assertEqual(start_column, 26)

        end_column = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[0].end_column
        self.assertEqual(end_column, 58)

        # second result location
        start_line = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[1].start_line
        self.assertEqual(start_line, 7)

        end_line = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[1].end_line
        self.assertEqual(end_line, 8)

        start_column = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[1].start_column
        self.assertEqual(start_column, 26)

        end_column = self.merged_query_dataset.query_and_files_results[0].\
            resultlocation[1].end_column
        self.assertEqual(end_column, 46)

    def test_realfile_raw_file(self):
        source = self.merged_query_dataset.query_and_files_results[2].raw_file_path.\
            file_path.dataset_file_path.source_name
        self.assertEqual(source, "eth_py150_open")

        split = self.merged_query_dataset.query_and_files_results[2].raw_file_path.\
            file_path.dataset_file_path.split
        self.assertEqual(split, 3)

        path = self.merged_query_dataset.query_and_files_results[2].raw_file_path.\
            file_path.dataset_file_path.unique_file_path
        self.assertEqual(
            path,
            "CGATOxford/cgat/CGAT/CSV.py")

        language = self.merged_query_dataset.query_and_files_results[2].raw_file_path.\
            language
        self.assertEqual(language, 0)

        content = self.merged_query_dataset.query_and_files_results[2].raw_file_path.file_content
        with open('./contexts/test_data/test_file_content.py', 'r') as f:
            actual_content = f.read()
        self.assertEqual(content, actual_content)

    def test_realfile_raw_query(self):
        repo = self.merged_query_dataset.query_and_files_results[2].query.query_path.repo
        self.assertEqual(repo, "https://github.com/github/codeql")

        path = self.merged_query_dataset.query_and_files_results[2].query.query_path.\
            unique_path
        self.assertEqual(
            path,
            "https://api.github.com/repos/github/codeql/git/blobs/ed4a240ec4d1066dfe008d4e19a66ef6bb187b46"
        )

        actual_id = bytes(
            "\223FK\311MECUZ:^\000I\213\256\"", "utf-8"
        )
        id = self.merged_query_dataset.query_and_files_results[2].query.queryID
        self.assertEqual(id, actual_id)

        actual_content = "/**\n * @name `__iter__` method returns a non-iterator\n * \
@description The `__iter__` method returns a non-iterator which, if used in a \'for\' loop, would raise a \'TypeError\'.\n * \
@kind problem\n * @tags reliability\n *       correctness\n * @problem.severity error\n * @sub-severity low\n * \
@precision high\n * @id py/iter-returns-non-iterator\n */\n\nimport python\n\n\
from ClassValue iterable, FunctionValue iter, ClassValue iterator\nwhere\n  iter = iterable.lookup(\"__iter__\") \
and\n  iterator = iter.getAnInferredReturnType() and\n  not iterator.isIterator()\nselect iterator,\
\n  \"Class \" + iterator.getName() +\n    \" is returned as an iterator (by $@) but does not fully \
implement the iterator interface.\",\n  iter, iter.getName()\n"

        content = self.merged_query_dataset.query_and_files_results[2].query.content
        self.assertEqual(actual_content, content)

        actual_description = "The `__iter__` method returns a non-iterator which, if used in a 'for' loop, would raise a 'TypeError'."
        description = self.merged_query_dataset.query_and_files_results[2].\
            query.metadata.description
        self.assertEqual(actual_description, description)

        actual_severity = "error\\n"
        severity = self.merged_query_dataset.query_and_files_results[2].\
            query.metadata.severity
        self.assertEqual(actual_severity, severity)

        actual_message = "NA"
        message = self.merged_query_dataset.query_and_files_results[2].\
            query.metadata.message
        self.assertEqual(actual_message, message)

        language = self.merged_query_dataset.query_and_files_results[2].\
            query.language
        self.assertEqual(language, 0)

        hop = self.merged_query_dataset.query_and_files_results[2].query.distributable
        self.assertEqual(False, hop)

        span = self.merged_query_dataset.query_and_files_results[2].\
            query.span
        self.assertEqual(1, span)

    def test_realfile_result_location(self):
        start_line = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].start_line
        self.assertEqual(start_line, 69)

        end_line = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].end_line
        self.assertEqual(end_line, 69)

        start_column = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].start_column
        self.assertEqual(start_column, 1)

        end_column = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].end_column
        self.assertEqual(end_column, 22)
        start_line = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].supporting_fact_locations[0].start_line
        self.assertEqual(start_line, 79)

        end_line = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].supporting_fact_locations[0].end_line
        self.assertEqual(end_line, 79)

        start_column = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].supporting_fact_locations[0].start_column
        self.assertEqual(start_column, 5)

        end_column = self.merged_query_dataset.query_and_files_results[2].\
            resultlocation[0].supporting_fact_locations[0].end_column
        self.assertEqual(end_column, 23)


if __name__ == "__main__":
    unittest.main()
