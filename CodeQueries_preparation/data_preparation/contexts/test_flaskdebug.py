from get_context import get_span_context
from basecontexts import MODULE_FUNCTION, Block, MODULE_OTHER
from tree_sitter import Language, Parser
from collections import namedtuple
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

# train_data: /peterhudec/authomatic/tests/functional_tests/expected_values/tumblr.py
# '''from __future__ import absolute_import
# from flask import Flask, request

# app = Flask(__name__)

# def setup():
#     with open(SECRET_FILE, 'w') as f:
#         f.write(''.join(utils.random_string(size=42)))

# setup()
# with open(SECRET_FILE) as f:
#     app.secret_key = f.read()

# @app.route('/'+app.secret_key, methods=['POST'])
# def main():
#     return "OK"

# def start(app):
#     app.run('0.0.0.0', port=5678, debug=True)

# if __name__ == "__main__":
#     start(app)'''

SOURCE_CODE = ['''from __future__ import absolute_import\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\ndef setup():\n    with open(SECRET_FILE, \'w\') as f:\n        f.write(\'\'.join(utils.random_string(size=42)))\n\nsetup()\nwith open(SECRET_FILE) as f:\n    app.secret_key = f.read()\n\n@app.route(\'/\'+app.secret_key, methods=[\'POST\'])\ndef main():\n    return "OK"\n\ndef start(app):\n    app.run(\'0.0.0.0\', port=5678, debug=True)\n\nif __name__ == "__main__":\n    start(app)''']
SPANS = [Span(18, 4, 18, 65)]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(17,
                            18,
                            [],
                            '''def start(app):\n    app.run('0.0.0.0', port=5678, debug=True)''',
                            'root.start',
                            MODULE_FUNCTION,
                            True,
                            'module',
                            ('__', '__class__')),
                      Block(0,
                            21,
                            [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 19, 20, 21],
                            '''from __future__ import absolute_import\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\n\nsetup()\nwith open(SECRET_FILE) as f:\n    app.secret_key = f.read()\n\n\n\nif __name__ == "__main__":\n    start(app)''',
                            'root',
                            MODULE_OTHER,
                            True,
                            'module',
                            ('__', '__class__'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPANS[i]
            generated_block = get_span_context('Flask app is run in debug mode',
                                               code, tree_sitter_parser, '',
                                               '', span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
