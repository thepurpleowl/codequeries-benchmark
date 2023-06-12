import unittest
import csv
import tempfile
import dataset_with_context_pb2
from tree_sitter import Language, Parser
from create_query_result import create_query_result, create_query_result_merged
from create_tokenized_files_labels import create_tokenized_files_labels
from create_blocks_labels_dataset import create_blocks_labels_dataset
from create_blocks_relevance_labels_dataset import create_blocks_relevance_labels_dataset

CSV_FIELDS_RESULTS = ["Name", "Description", "Warning", "Severity",
                      "Path", "Start_line", "Start_column", "End_line",
                      "End_column"]

CSV_ENTRIES_RESULTS = [
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
QUERY_NAME_TOKENS = [['`', '__iter__', '`', 'method', 'returns', 'a', 'non', '-', 'iterator'], ['Unreachable', 'code']]

program_dataset = dataset_with_context_pb2.RawProgramDataset()
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


class TestCreateBlockLabelsDataset(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    PY_LANGUAGE = Language("./my-languages.so", "python")

    tree_sitter_parser = Parser()
    tree_sitter_parser.set_language(PY_LANGUAGE)

    dataset = create_query_result(program_dataset, query_dataset,
                                  results_file, 'positive')
    dataset_merged = create_query_result_merged(dataset)

    tokenized_files_with_labels = create_tokenized_files_labels(dataset_merged, 'positive')
    tokenized_files_with_labels.example_type = dataset_with_context_pb2.ExampleType.Value('positive')

    # results_file is passed as aux_results file as these two queries
    # don't need auxilliary results
    tokenized_block_labels_dataset = create_blocks_labels_dataset(
        tokenized_files_with_labels,
        tree_sitter_parser, True, None)

    blockdata_instance_to_check = tokenized_block_labels_dataset.tokenized_block_query_labels_item[0]

    # results_file is passed as aux_results file as these two queries
    # don't need auxilliary results
    only_relevant_tokenized_block_labels_dataset = create_blocks_relevance_labels_dataset(
        tokenized_files_with_labels, tree_sitter_parser,
        True, results_file, 'yes')

    only_relevant_blockdata_instance_to_check = only_relevant_tokenized_block_labels_dataset.tokenized_block_query_labels_item[0]

    # print(len(only_relevant_tokenized_block_labels_dataset.tokenized_block_query_labels_item))
    # for y in only_relevant_tokenized_block_labels_dataset.tokenized_block_query_labels_item[0].tokenized_block_query_labels_group_item:
    #     print(y.block)
    #     print('-' * 100)

    def test_query_details(self):
        for block_data in self.blockdata_instance_to_check.tokenized_block_query_labels_group_item:
            self.assertEqual(block_data.query_id,
                             bytes("\223FK\311MECUZ:^\000I\213\256\"", "utf-8"))

            self.assertEqual(block_data.distributable,
                             False)

            self.assertEqual(block_data.query_name_tokens,
                             QUERY_NAME_TOKENS[0])

    def test_raw_file_details(self):
        for block_data in self.blockdata_instance_to_check.tokenized_block_query_labels_group_item:
            self.assertEqual(block_data.raw_file,
                             program_dataset.raw_program_dataset[0])

    def test_block_details(self):
        block1 = dataset_with_context_pb2.ProgramBlockDetails()
        block1.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_FUNCTION")
        block1.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block1.start_line = 78
        block1.end_line = 79
        block1.other_lines.extend([])
        block1.unique_block_id = b'&{(\x89\xe8[{\x01~J\xa4\xf5\x93\x81\xbc\xb8'
        block1.content = "def __iter__(self):\n        return self"
        block1.metadata = "root.CommentStripper.__iter__"
        block1.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block2 = dataset_with_context_pb2.ProgramBlockDetails()
        block2.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_OTHER")
        block2.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block2.start_line = 68
        block2.end_line = 87
        block2.other_lines.extend([68, 69, 70, 71, 72, 73, 74, 77, 80])
        block2.unique_block_id = b'\x0b\x1a\xaf\xcb\xd1.\xf8\xb2#\xd5{W\xd0\x86\x94\x8c'
        block2.content = "class CommentStripper:\n    \"\"\"Iterator for stripping comments \
from file.\n\n    This iterator will skip any lines beginning with ``#``\n    \
or any empty lines at the beginning of the output.\n    \"\"\"\n\n\n"
        block2.metadata = "root.CommentStripper"
        block2.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        blocks = [block1, block2]

        for i, block_data in enumerate(self.blockdata_instance_to_check.tokenized_block_query_labels_group_item):
            self.assertEqual(block_data.block, blocks[i])

    def test_only_relevant_block_details(self):
        block1 = dataset_with_context_pb2.ProgramBlockDetails()
        block1.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_FUNCTION")
        block1.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block1.start_line = 78
        block1.end_line = 79
        block1.other_lines.extend([])
        block1.unique_block_id = b'&{(\x89\xe8[{\x01~J\xa4\xf5\x93\x81\xbc\xb8'
        block1.content = "def __iter__(self):\n        return self"
        block1.metadata = "root.CommentStripper.__iter__"
        block1.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block2 = dataset_with_context_pb2.ProgramBlockDetails()
        block2.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_OTHER")
        block2.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block2.start_line = 68
        block2.end_line = 87
        block2.other_lines.extend([68, 69, 70, 71, 72, 73, 74, 77, 80])
        block2.unique_block_id = b'\x0b\x1a\xaf\xcb\xd1.\xf8\xb2#\xd5{W\xd0\x86\x94\x8c'
        block2.content = "class CommentStripper:\n    \"\"\"Iterator for stripping comments \
from file.\n\n    This iterator will skip any lines beginning with ``#``\n    \
or any empty lines at the beginning of the output.\n    \"\"\"\n\n\n"
        block2.metadata = "root.CommentStripper"
        block2.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block3 = dataset_with_context_pb2.ProgramBlockDetails()
        block3.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_FUNCTION")
        block3.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block3.start_line = 110
        block3.end_line = 111
        block3.other_lines.extend([])
        block3.unique_block_id = b'\x1b0T?\x10\x91g\xbe\x8b\xe2\xf7i\xf9\t\x81A'
        block3.content = "def __iter__(self):\n        return self"
        block3.metadata = "root.UnicodeCsvReader.__iter__"
        block3.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block4 = dataset_with_context_pb2.ProgramBlockDetails()
        block4.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_FUNCTION")
        block4.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block4.start_line = 147
        block4.end_line = 148
        block4.other_lines.extend([])
        block4.unique_block_id = b'V|\x90\x0f\xb6\x95\x95m\x8d\xa1\xcc\x9d\x83\x8ee('
        block4.content = "def __iter__(self):\n        return self"
        block4.metadata = "root.DictReaderLarge.__iter__"
        block4.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block5 = dataset_with_context_pb2.ProgramBlockDetails()
        block5.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_OTHER")
        block5.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block5.start_line = 104
        block5.end_line = 121
        block5.other_lines.extend([104, 105, 109, 112, 118])
        block5.unique_block_id = b'\x98\x0ewf\xd9\x8b\xe4\x86e\xc7\x9a\x89\x19\xd0\x00"'
        block5.content = "class UnicodeCsvReader(object):\n\n\n\n"
        block5.metadata = "root.UnicodeCsvReader"
        block5.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block6 = dataset_with_context_pb2.ProgramBlockDetails()
        block6.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("CLASS_OTHER")
        block6.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block6.start_line = 131
        block6.end_line = 157
        block6.other_lines.extend([131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 146, 149])
        block6.unique_block_id = b'${\xb1\xd4\\\x08\xaf\xf3\xb9\x02\x9b\x82<wC;'
        block6.content = "class DictReaderLarge:\n    \"\"\"Substitute for :py:class:`csv.DictReader` that handles very large\n    \
fields.\n\n    :py:mod:`csv` is implemented in C and limits the number of columns\n    per table. This class has no such limit, but will \
not be as fast.\n\n    This class is only a minimal implementation. For example, it does\n    not handle dialects.\n    \"\"\"\n\n\n"
        block6.metadata = "root.DictReaderLarge"
        block6.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        blocks = [block1, block2, block3, block4, block5, block6]

        for i, block_data in enumerate(self.only_relevant_blockdata_instance_to_check.tokenized_block_query_labels_group_item):
            self.assertEqual(block_data.block, blocks[i])


if __name__ == "__main__":
    unittest.main()
