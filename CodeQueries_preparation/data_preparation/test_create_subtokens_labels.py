import unittest
import csv
import tempfile
import dataset_with_context_pb2
from tree_sitter import Language, Parser
from create_query_result import create_query_result, create_query_result_merged
from create_tokenized_files_labels import create_tokenized_files_labels
from create_blocks_labels_dataset import create_blocks_labels_dataset
from create_blocks_relevance_labels_dataset import create_blocks_relevance_labels_dataset
from create_block_subtokens_labels import create_cubert_subtokens_labels

SUBTOKENS = ['module_', '\\u\\u\\uEOS\\u\\u\\u_', '\\u\\u\\uNL\\u\\u\\u_', 'def_', 'purchase', '_', '(_',
             'self_', ',_', 'amount_', ',_', 'options_', '=_', 'None_', ')_',
             ':_', '\\u\\u\\uNEWLINE\\u\\u\\u_', '\\u\\u\\uINDENT\\u\\u\\u ', '   _', 'if_', 'not_', 'options_', ':_',
             '\\u\\u\\uNEWLINE\\u\\u\\u_', '\\u\\u\\uINDENT\\u\\u\\u ', '       _', 'options_', '=_',
             '{_', '}_', '\\u\\u\\uNEWLINE\\u\\u\\u_', '\\u\\u\\uDEDENT\\u\\u\\u_',
             'tmp', '\\u', 'options_', '=_', 'options_', '._', 'copy_', '(_', ')_', '\\u\\u\\uNEWLINE\\u\\u\\u_',
             'permissi', 'ble', '\\u', 'options_', '=_', '[_', '"', 'Sen', 'der', 'Token', 'Id', '"_', ',_', '"',
             'Caller', 'Reference', '"_', ',_', '\\u\\u\\uNL\\u\\u\\u_', '"', 'Sen', 'der', 'Descripti', 'on', '"_',
             ',_', '"', 'Caller', 'Descripti', 'on', '"_', ',_', '"', 'Transa', 'ction', 'Time', 'out', 'In', 'Min', 's',
             '"_', '\\u\\u\\uNL\\u\\u\\u_', '"', 'Transa', 'ction', 'Amo', 'unt', '"_', ',_', '"', 'Override', 'IP',
             'NU', 'RL', '"_', ',_', '"', 'Descrip', 'tor', 'Polic', 'y', '"_', ']_', '\\u\\u\\uNEWLINE\\u\\u\\u_']

LABELS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

SECOND_BLOCK_SUBTOKENS = ['module_', '\\u\\u\\uEOS\\u\\u\\u_', '\\u\\u\\uNL\\u\\u\\u_', '\\u\\u\\uDEDENT\\u\\u\\u_', 'def_', 'sell', '_', '(_',
                          'self_', ',_', 'amount_', ',_', 'options_', '=_', 'None_', ')_',
                          ':_', '\\u\\u\\uNEWLINE\\u\\u\\u_', '\\u\\u\\uINDENT\\u\\u\\u ', '   _', 'if_', 'not_', 'options_', ':_',
                          '\\u\\u\\uNEWLINE\\u\\u\\u_', '\\u\\u\\uINDENT\\u\\u\\u ', '       _', 'options_', '=_',
                          '{_', '}_', '\\u\\u\\uNEWLINE\\u\\u\\u_', '\\u\\u\\uDEDENT\\u\\u\\u_',
                          'tmp', '\\u', 'options_', '=_', 'options_', '._', 'copy_', '(_', ')_', '\\u\\u\\uNEWLINE\\u\\u\\u_',
                          'permissi', 'ble', '\\u', 'options_', '=_', '[_', '"', 'Sen', 'der', 'Token', 'Id', '"_', ',_', '"',
                          'Caller', 'Reference', '"_', ',_', '\\u\\u\\uNL\\u\\u\\u_', '"', 'Sen', 'der', 'Descripti', 'on', '"_',
                          ',_', '"', 'Caller', 'Descripti', 'on', '"_', ',_', '"', 'Transa', 'ction', 'Time', 'out', 'In', 'Min', 's',
                          '"_', '\\u\\u\\uNL\\u\\u\\u_', '"', 'Transa', 'ction', 'Amo', 'unt', '"_', ',_', '"', 'Override', 'IP',
                          'NU', 'RL', '"_', ',_', '"', 'Descrip', 'tor', 'Polic', 'y', '"_', ']_']

SECOND_BLOCK_LABELS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

QUERY_NAME_SUBTOKENS = ["Implicit", "_", "string_", "concate", "nation_", "in_", "a_", "list_"]

CSV_FIELDS_RESULTS = ["Name", "Description", "Warning", "Severity",
                      "Path", "Start_line", "Start_column", "End_line",
                      "End_column"]

CSV_ENTRIES_RESULTS = [
    ["Implicit string concatenation in a list",
     "Omitting a comma between strings causes implicit concatenation which \
is confusing in a list.",
     "warning",
     "Implicit string concatenation. Maybe missing a comma?",
     "/agiliq/merchant/billing/integrations/amazon_fps_integration.py",
     9, 51, 10, 27],
    ["Implicit string concatenation in a list",
     "Omitting a comma between strings causes implicit concatenation which \
is confusing in a list.",
     "warning",
     "Implicit string concatenation. Maybe missing a comma?",
     "/agiliq/merchant/billing/integrations/amazon_fps_integration.py",
     17, 51, 18, 27],
]

program_dataset = dataset_with_context_pb2.RawProgramDataset()
program = dataset_with_context_pb2.RawProgramFile()
program.file_path.dataset_file_path.source_name = ("eth_py150_open")
program.file_path.dataset_file_path.split = 3
program.file_path.dataset_file_path.unique_file_path = (
    "agiliq/merchant/billing/integrations/amazon_fps_integration.py")
program.language = 0
program.file_content = """from billing.integration import Integration, IntegrationNotConfigured
from django.conf import settings

def purchase(self, amount, options=None):
    if not options:
        options = {}
    tmp_options = options.copy()
    permissible_options = ["SenderTokenId", "CallerReference",
        "SenderDescription", "CallerDescription", "TransactionTimeoutInMins"
        "TransactionAmount", "OverrideIPNURL", "DescriptorPolicy"]

def sell(self, amount, options=None):
    if not options:
        options = {}
    tmp_options = options.copy()
    permissible_options = ["SenderTokenId", "CallerReference",
        "SenderDescription", "CallerDescription", "TransactionTimeoutInMins"
        "TransactionAmount", "OverrideIPNURL", "DescriptorPolicy"]"""

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
query.distributable = True
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


class TestCreateBlockSubtokenDataset(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    PY_LANGUAGE = Language("./my-languages.so", "python")

    tree_sitter_parser = Parser()
    tree_sitter_parser.set_language(PY_LANGUAGE)

    query_dataset = create_query_result(program_dataset, query_dataset,
                                        results_file, 'positive')
    dataset_merged = create_query_result_merged(query_dataset)

    tokenized_files_with_labels = create_tokenized_files_labels(dataset_merged, 'positive')
    tokenized_files_with_labels.example_type = dataset_with_context_pb2.ExampleType.Value('positive')

    tokenized_block_query_labels_dataset = create_blocks_labels_dataset(
        tokenized_files_with_labels,
        tree_sitter_parser, True, None)

    dataset = create_cubert_subtokens_labels("line_number", tokenized_block_query_labels_dataset,
                                             "vocab.txt")

    subtoken_instance_to_check = dataset.block_query_subtokens_labels_item[0]

    def test_query_details(self):
        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].query_id,
                         bytes("\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8"))

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].distributable,
                         True)

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].query_name_subtokens,
                         QUERY_NAME_SUBTOKENS)

    def test_raw_file(self):
        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].raw_file,
                         program_dataset.raw_program_dataset[0])

    def test_block_data(self):
        block1 = dataset_with_context_pb2.ProgramBlockDetails()
        block1.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("MODULE_FUNCTION")
        block1.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block1.start_line = 3
        block1.end_line = 9
        block1.other_lines.extend([])
        block1.unique_block_id = b'\xec\x0fM$ESV\x88b\xc3[\x8c\xca\xe4\x90"'
        block1.content = "def purchase(self, amount, options=None):\n    if not options:\n        options = {}\n    \
tmp_options = options.copy()\n    permissible_options = [\"SenderTokenId\", \"CallerReference\",\n        \"SenderDescription\", \
\"CallerDescription\", \"TransactionTimeoutInMins\"\n        \"TransactionAmount\", \"OverrideIPNURL\", \"DescriptorPolicy\"]"
        block1.metadata = "root.purchase"
        block1.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block2 = dataset_with_context_pb2.ProgramBlockDetails()
        block2.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("MODULE_FUNCTION")
        block2.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block2.start_line = 11
        block2.end_line = 17
        block2.other_lines.extend([])
        block2.unique_block_id = b'Ho\x9b\xd6\x07\x1f\xd3]\xf4\x85t\xba^ :\x1a'
        block2.content = "def sell(self, amount, options=None):\n    if not options:\n        options = {}\n    \
tmp_options = options.copy()\n    permissible_options = [\"SenderTokenId\", \"CallerReference\",\n        \"SenderDescription\", \
\"CallerDescription\", \"TransactionTimeoutInMins\"\n        \"TransactionAmount\", \"OverrideIPNURL\", \"DescriptorPolicy\"]"
        block2.metadata = "root.sell"
        block2.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block,
                         block1)

        self.assertEqual(self.dataset.block_query_subtokens_labels_item[1].block_query_subtokens_labels_group_item[0].block,
                         block2)

        # print(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item)
        self.assertEqual(len(self.dataset.block_query_subtokens_labels_item),
                         2)
        self.assertEqual(len(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item),
                         1)

    def test_subtokens(self):
        for i in range(len(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block_subtokens_labels)):
            self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block_subtokens_labels[i].
                             program_subtoken,
                             SUBTOKENS[i])

            self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block_subtokens_labels[i].
                             label,
                             LABELS[i])


class TestCreateBlockLabelsDataset(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    PY_LANGUAGE = Language("./my-languages.so", "python")

    tree_sitter_parser = Parser()
    tree_sitter_parser.set_language(PY_LANGUAGE)

    query_dataset = create_query_result(program_dataset, query_dataset,
                                        results_file, 'positive')
    dataset_merged = create_query_result_merged(query_dataset)

    tokenized_files_with_labels = create_tokenized_files_labels(dataset_merged, 'positive')

    tokenized_block_query_labels_dataset = create_blocks_relevance_labels_dataset(
        tokenized_files_with_labels,
        tree_sitter_parser, True, None, 'yes')

    dataset = create_cubert_subtokens_labels("line_number", tokenized_block_query_labels_dataset,
                                             "vocab.txt")

    subtoken_instance_to_check = dataset.block_query_subtokens_labels_item[0]

    def test_query_details(self):
        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].query_id,
                         bytes("\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8"))

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].distributable,
                         True)

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].query_name_subtokens,
                         QUERY_NAME_SUBTOKENS)

    def test_raw_file(self):
        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].raw_file,
                         program_dataset.raw_program_dataset[0])

    def test_block_data(self):
        block1 = dataset_with_context_pb2.ProgramBlockDetails()
        block1.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("MODULE_FUNCTION")
        block1.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block1.start_line = 3
        block1.end_line = 9
        block1.other_lines.extend([])
        block1.unique_block_id = b'\xec\x0fM$ESV\x88b\xc3[\x8c\xca\xe4\x90"'
        block1.content = "def purchase(self, amount, options=None):\n    if not options:\n        options = {}\n    \
tmp_options = options.copy()\n    permissible_options = [\"SenderTokenId\", \"CallerReference\",\n        \"SenderDescription\", \
\"CallerDescription\", \"TransactionTimeoutInMins\"\n        \"TransactionAmount\", \"OverrideIPNURL\", \"DescriptorPolicy\"]"
        block1.metadata = "root.purchase"
        block1.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        block2 = dataset_with_context_pb2.ProgramBlockDetails()
        block2.block_type = dataset_with_context_pb2.ProgramBlockTypes.Value("MODULE_FUNCTION")
        block2.relevance_label = dataset_with_context_pb2.BlockRelevance.Value("yes")
        block2.start_line = 11
        block2.end_line = 17
        block2.other_lines.extend([])
        block2.unique_block_id = b'Ho\x9b\xd6\x07\x1f\xd3]\xf4\x85t\xba^ :\x1a'
        block2.content = "def sell(self, amount, options=None):\n    if not options:\n        options = {}\n    \
tmp_options = options.copy()\n    permissible_options = [\"SenderTokenId\", \"CallerReference\",\n        \"SenderDescription\", \
\"CallerDescription\", \"TransactionTimeoutInMins\"\n        \"TransactionAmount\", \"OverrideIPNURL\", \"DescriptorPolicy\"]"
        block2.metadata = "root.sell"
        block2.file_path.CopyFrom(program_dataset.raw_program_dataset[0].file_path)

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block,
                         block1)

        self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[1].block,
                         block2)

        # print(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item)
        self.assertEqual(len(self.dataset.block_query_subtokens_labels_item),
                         1)
        self.assertEqual(len(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item),
                         2)

    def test_subtokens(self):
        for i in range(len(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block_subtokens_labels)):
            self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block_subtokens_labels[i].
                             program_subtoken,
                             SUBTOKENS[i])

            self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[0].block_subtokens_labels[i].
                             label,
                             LABELS[i])

        for i in range(len(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[1].block_subtokens_labels)):
            self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[1].block_subtokens_labels[i].
                             program_subtoken,
                             SECOND_BLOCK_SUBTOKENS[i])

            self.assertEqual(self.subtoken_instance_to_check.block_query_subtokens_labels_group_item[1].block_subtokens_labels[i].
                             label,
                             SECOND_BLOCK_LABELS[i])


if __name__ == "__main__":
    unittest.main()
