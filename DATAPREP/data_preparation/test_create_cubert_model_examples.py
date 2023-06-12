import unittest
import csv
import tempfile
import dataset_with_context_pb2
from tree_sitter import Language, Parser
from create_query_result import create_query_result
from create_query_result import create_query_result_merged
from create_tokenized_files_labels import create_tokenized_files_labels
from create_blocks_labels_dataset import create_blocks_labels_dataset
from create_block_subtokens_labels import create_cubert_subtokens_labels
from create_single_model_baseline_examples import create_single_model_examples

INPUT_IDS = [2, 46114, 26, 354, 48016, 41129, 58, 208, 141, 3, 274, 144, 10, 39, 25740, 26, 21, 28, 18, 2699, 18, 304,
             22, 61, 20, 24, 15, 29, 60, 41, 96, 304, 24, 15, 29, 40, 304, 22, 63, 62, 15, 7, 5664, 4395, 304, 22, 304,
             19, 721, 21, 20, 15, 45503, 7486, 4395, 304, 22, 30, 49520, 11635, 7155, 22698, 9470, 36, 18, 49520, 33705,
             25311, 36, 18, 10, 49520, 11635, 7155, 39840, 7970, 36, 18, 49520, 33705, 39840, 7970, 36, 18, 49520, 43000,
             31428, 11737, 5133, 6443, 10764, 15096, 36, 10, 49520, 43000, 31428, 41711, 13475, 36, 18, 49520, 33086,
             6906, 11794, 9651, 36, 18, 49520, 35697, 10533, 46407, 15934, 36, 31]
INPUT_MASK = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1]
SEGMENT_IDS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1]
LABEL_IDS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2]

BLOCK_IDS = [b'\xec\x0fM$ESV\x88b\xc3[\x8c\xca\xe4\x90"']

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
"""

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


class TestCreateModelExamples(unittest.TestCase):
    results_file, results_csv = generate_csv_file_no_column_names(
        CSV_ENTRIES_RESULTS)

    PY_LANGUAGE = Language("./my-languages.so", "python")

    tree_sitter_parser = Parser()
    tree_sitter_parser.set_language(PY_LANGUAGE)

    dataset = dataset_with_context_pb2.RawResultDataset()

    dataset = create_query_result(program_dataset, query_dataset,
                                  results_file, 'positive')
    dataset_merged = create_query_result_merged(dataset)

    tokenized_files_with_labels = create_tokenized_files_labels(dataset_merged, 'positive')
    tokenized_files_with_labels.example_type = dataset_with_context_pb2.ExampleType.Value('positive')

    tokenized_block_query_labels_dataset = create_blocks_labels_dataset(
        tokenized_files_with_labels,
        tree_sitter_parser, True, None)

    dataset = create_cubert_subtokens_labels("line_number", tokenized_block_query_labels_dataset,
                                             "vocab.txt")

    examples_dataset = create_single_model_examples(dataset, "vocab.txt", 'cubert')

    def test_query_details(self):
        self.assertEqual(self.examples_dataset.examples[0].query_id,
                         bytes("\276F\034Om\351G\373\031\362\206\255\n\007E\321", "utf-8"))

        self.assertEqual(self.examples_dataset.examples[0].distributable,
                         True)

        self.assertEqual(self.examples_dataset.examples[0].example_type,
                         dataset_with_context_pb2.ExampleType.Value('positive'))

    def test_block_ids(self):
        assert len(self.examples_dataset.examples) == 1
        for j in range(len(self.examples_dataset.examples[0].block_id)):
            self.assertEqual(self.examples_dataset.examples[0].block_id[j],
                             BLOCK_IDS[j])

    def test_input_ids(self):
        for j in range(len(self.examples_dataset.examples[0].input_ids)):
            self.assertEqual(self.examples_dataset.examples[0].input_ids[j],
                             INPUT_IDS[j])

    def test_input_mask(self):
        for j in range(len(self.examples_dataset.examples[0].input_mask)):
            self.assertEqual(self.examples_dataset.examples[0].input_mask[j],
                             INPUT_MASK[j])

    def test_label_ids(self):
        for j in range(len(self.examples_dataset.examples[0].labels_ids)):
            self.assertEqual(self.examples_dataset.examples[0].labels_ids[j],
                             LABEL_IDS[j])

    def test_segment_ids(self):
        for j in range(len(self.examples_dataset.examples[0].segment_ids)):
            self.assertEqual(self.examples_dataset.examples[0].segment_ids[j],
                             SEGMENT_IDS[j])


if __name__ == "__main__":
    unittest.main()
