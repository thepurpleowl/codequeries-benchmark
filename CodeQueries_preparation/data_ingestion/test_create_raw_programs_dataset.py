import unittest
import create_raw_programs_dataset
from collections import namedtuple

LANGUAGE = 0
CONTENTS = [[""],
            ["from __future__ import unicode_literals",
             "from ctypes import windll, Structure, c_uint",
             "from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL", "",
             "shell32 = windll.shell32",
             "SHFileOperationW = shell32.SHFileOperationW", "", "",
             "class SHFILEOPSTRUCTW(Structure):",
             "    _fields_ = [", "        ('hwnd, HWND),",
             "        ('wFunc, UINT),", "        ('pFrom, LPCWSTR),",
             "        ('pTo, LPCWSTR),", "        ('fFlags, c_uint),",
             "        ('fAnyOperationsAborted, BOOL),",
             "        ('hNameMappings, c_uint),",
             "        ('lpszProgressTitle, LPCWSTR),", "    ]", "", "",
             "FO_MOVE = 1", "FO_COPY = 2", "FO_DELETE = 3", "FO_RENAME = 4",
             ""]]

SOURCE_NAME = "Test Dataset"
SPLIT = 3
PATH = ["empty_file.py", "plat_win.py"]


class TestCreateRawProgramsDatasetNonGithub(unittest.TestCase):
    """
    This tests all functionalities when creating dataset using
    Python files taken from sources other than GitHub.
    """

    def setUp(self):
        File_Path_Content = namedtuple("File_Path_Content", "path content")
        files = ["empty_file.py", "plat_win.py"]

        file_content = []
        for i in range(len(files)):
            temp = File_Path_Content(files[i], "\n".join(
                k for k in CONTENTS[i]))
            file_content.append(temp)

        self.returned_data = (
            create_raw_programs_dataset.CreateRawProgramsDatasetNonGithub(
                "Test Dataset",
                "VALIDATION",
                "Python",
                file_content

            ))

    # Test if the programming language information is stored correctly
    def test_language(self):
        for i in range(len(self.returned_data.raw_program_dataset)):
            self.assertEqual(
                self.returned_data.raw_program_dataset[i].language, LANGUAGE)

    # Test if the file contents are stored correctly.
    def test_file_content(self):
        for i in range(len(self.returned_data.raw_program_dataset)):
            returned_content = (
                self.returned_data.raw_program_dataset[i].file_content)
            actual_content = "\n".join(j for j in CONTENTS[i])
            self.assertEqual(returned_content, actual_content)

    # Test if the file paths are stored correctly.
    def test_file_path(self):
        for i in range(len(self.returned_data.raw_program_dataset)):
            source = (
                self.returned_data.raw_program_dataset[i]
                    .file_path.dataset_file_path.source_name)
            split_name = self.returned_data.raw_program_dataset[i]\
                .file_path.dataset_file_path.split
            unique_path = self.returned_data.raw_program_dataset[
                i].file_path.dataset_file_path.unique_file_path

            self.assertEqual(source, SOURCE_NAME)
            self.assertEqual(split_name, SPLIT)
            self.assertEqual(unique_path, PATH[i])


if __name__ == "__main__":
    unittest.main()
