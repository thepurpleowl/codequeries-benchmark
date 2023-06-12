import raw_codeql_queryset_pb2 as raw_codeql_queryset_pb2
import create_raw_codeql_queryset
import unittest
import csv
import tempfile
import os.path


CSV_FIELDS = ["Path", "Url", "Description", "Hops", "Span"]
CSV_ENTRIES = [["query_files/test_query_0.ql", "https://github.com/thepurpleowl/test_case_repo/blob/main/query_files/test_query_0.ql", "Empty query", "1", "k"],
               ["query_files/test_query_1.ql", "https://github.com/thepurpleowl/test_case_repo/blob/main/query_files/test_query_1.ql", "Query with only metadata", "k", "k"],
               ["query_files/test_query_2.ql", "https://github.com/thepurpleowl/test_case_repo/blob/main/query_files/test_query_2.ql", "Query with only metadata but with missing fields in metadata", "1", "1"],
               ["query_files/test_query_3.ql", "https://github.com/thepurpleowl/test_case_repo/blob/main/query_files/test_query_3.ql", "Query to check escape character", "1", "1"]]

QUERY_FILES = ["test_query_0.ql", "test_query_1.ql", "test_query_2.ql", "test_query_3.ql"]
REPO = "https://github.com/thepurpleowl/test_case_repo"
UNIQUE_PATHS = ["https://api.github.com/repos/thepurpleowl/test_case_repo/git/blobs/8b137891791fe96927ad78e64b0aad7bded08bdc",
                "https://api.github.com/repos/thepurpleowl/test_case_repo/git/blobs/d8c8534f64b3748ac32201268868b38c49e66e49",
                "https://api.github.com/repos/thepurpleowl/test_case_repo/git/blobs/fddc8416d2d1b41e0aa8cc5284a9700840f381da",
                "https://api.github.com/repos/thepurpleowl/test_case_repo/git/blobs/cdbcec278dd7a340fff1bf2d7f4ad71a0b0db3f8"]
QUERY_ID = [b"\325\300h\262\324}\331\310\357\3066P\343\305\200\250",
            b"{+\225\356\376\026 n\227\270\253\030\353\005xt",
            b"\025\312%\2274\374\257eV\261\333\006a\305\273\016",
            b"\227rO<nt>v\r\333w\022j\327|j"]
CONTENT = ["\n",
           "/**\n * @id py/examples/backticks\n * @name String conversion expressions\n * @description Finds `String conversions` expressions (expressions enclosed in backticks), which are removed in Python 3\n * @kind problem\n * @problem.severity warning\n * @sub-severity high\n * @precision very-high\n * @tags backtick\n *       string conversion\n */\n",
           "/**\n * @id py/examples/backticks\n * @name Unnecessary \'else\' clause and also \'\\n\' in loop\n * @description Finds `String conversions` expressions (expressions enclosed in backticks), which are removed in Python 3\n * @tags backtick\n *       string conversion\n */\n\nimport python\n\nfrom Repr r\nselect \n",
           "/**\n * @name \'import *\' may pollute namespace\n * @description Importing a module using \'import *\' may unintentionally pollute the global\n *              namespace if the module does not define `__all__`\n * @kind problem\n * @tags maintainability\n *       modularity\n * @problem.severity recommendation\n * @sub-severity high\n * @precision very-high\n * @id py/polluting-import\n */\n\nimport python\n\npredicate import_star(ImportStar imp, ModuleValue exporter) {\n  exporter.importedAs(imp.getImportedModuleName())\n}\n\npredicate all_defined(ModuleValue exporter) {\n  exporter.isBuiltin()\n  or\n  exporter.getScope().(ImportTimeScope).definesName(\"__all__\")\n  or\n  exporter.getScope().getInitModule().(ImportTimeScope).definesName(\"__all__\")\n}\n\nfrom ImportStar imp, ModuleValue exporter\nwhere import_star(imp, exporter) and not all_defined(exporter) and not exporter.isAbsent()\nselect imp,\n  \"Import pollutes the enclosing namespace, as the imported module $@ does not define \'__all__\'.\",\n  exporter, exporter.getName()\n"]
METADATA_NAME = ["NA",
                 "String conversion expressions",
                 "Unnecessary \'else\' clause and also \'\\n\' in loop",
                 "\'import *\' may pollute namespace"]
METADATA_DESC = ["NA",
                 "Finds `String conversions` expressions (expressions enclosed in backticks), which are removed in Python 3",
                 "Finds `String conversions` expressions (expressions enclosed in backticks), which are removed in Python 3",
                 "Importing a module using \'import *\' may unintentionally pollute the global namespace if the module does not define `__all__`"]
METADATA_SEVERITY = ["NA", "warning", "NA", "recommendation"]
METADATA_MESSAGE = ["NA", "NA", "NA", "NA"]
METADATA_FULL_METADATA = ["",
                          "/**\n * @id py/examples/backticks\n * @name String conversion expressions\n * @description Finds `String conversions` expressions (expressions enclosed in backticks), which are removed in Python 3\n * @kind problem\n * @problem.severity warning\n * @sub-severity high\n * @precision very-high\n * @tags backtick\n *       string conversion\n */",
                          "/**\n * @id py/examples/backticks\n * @name Unnecessary \'else\' clause and also \'\\n\' in loop\n * @description Finds `String conversions` expressions (expressions enclosed in backticks), which are removed in Python 3\n * @tags backtick\n *       string conversion\n */",
                          "/**\n * @name \'import *\' may pollute namespace\n * @description Importing a module using \'import *\' may unintentionally pollute the global\n *              namespace if the module does not define `__all__`\n * @kind problem\n * @tags maintainability\n *       modularity\n * @problem.severity recommendation\n * @sub-severity high\n * @precision very-high\n * @id py/polluting-import\n */"]
LANGUAGE = "Python"
HOPS = ["single_hop", "multiple_hops", "single_hop", "single_hop"]
SPAN = ["multiple_spans", "multiple_spans", "single_span", "single_span"]
DISTRIBUTABLE = [True, True, True, True]
CORRECT_QUERY_SERIALIZATION = raw_codeql_queryset_pb2.RawQueryList()


def generate_correct_query_serialization():
    for i, query_file in enumerate(QUERY_FILES):
        file_path_on_github = raw_codeql_queryset_pb2.GitHubFilePath()
        file_path_on_github.repo = REPO
        file_path_on_github.unique_path = UNIQUE_PATHS[i]

        query_metadata = raw_codeql_queryset_pb2.QueryMetadata()
        query_metadata.name = METADATA_NAME[i]
        query_metadata.description = METADATA_DESC[i]
        query_metadata.severity = METADATA_SEVERITY[i]
        query_metadata.message = METADATA_MESSAGE[i]
        query_metadata.full_metadata = METADATA_FULL_METADATA[i]

        query = raw_codeql_queryset_pb2.Query()
        query.query_path.CopyFrom(file_path_on_github)
        query.queryID = QUERY_ID[i]
        query.content = CONTENT[i]
        query.metadata.CopyFrom(query_metadata)
        query.language = raw_codeql_queryset_pb2.Languages.Value("Python")
        query.hops = raw_codeql_queryset_pb2.Hops.Value(HOPS[i])
        query.span = raw_codeql_queryset_pb2.Span.Value(SPAN[i])
        query.distributable = DISTRIBUTABLE[i]

        CORRECT_QUERY_SERIALIZATION.raw_query_set.append(query)


def generate_csv_file():
    csv_file_obj = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv")
    filename = csv_file_obj.name

    # create a csv writer object and write
    csvwriter = csv.writer(csv_file_obj)
    csvwriter.writerow(CSV_FIELDS)
    csvwriter.writerows(CSV_ENTRIES)
    csv_file_obj.flush()

    return filename, csv_file_obj


class TestQuerySerialization(unittest.TestCase):
    test_labeled_queries_file_path, csv_file_obj = generate_csv_file()
    target_programming_language = "Python"
    test_github_auth = ":"
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    test_save_queryset_location = os.path.join(__location__, "searialized_queryset")

    def test_query_serialization(self):
        create_raw_codeql_queryset.serialize_queryset(
            self.test_labeled_queries_file_path,
            self.target_programming_language,
            self.test_github_auth,
            self.test_save_queryset_location
        )

        self.csv_file_obj.close()

        generated_serialization = ""
        with open(self.test_save_queryset_location, "rb") as fd:
            generated_serialization = fd.read()

        generate_correct_query_serialization()
        reference_serializations = CORRECT_QUERY_SERIALIZATION.SerializeToString()
        # print(generated_serialization)
        self.assertEqual(generated_serialization, reference_serializations)


if __name__ == "__main__":
    unittest.main()
