from get_context import get_span_context
from basecontexts import Block, CLASS_FUNCTION, CLASS_OTHER
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

# import threading

# class TCPServer(object):
#     def __init__(self, server):
#         self.port = 22
#         self.server = server

#     def process_request(self, request, client_address):
#         self.do_work(request, client_address)
#         self.shutdown_request(request)

#     def allows(self, request, client_address):
#         return threading.alows(client_address, request)

#     def server_allows(self, request):
#         return threading.alows(self.server, request)

# class ThreadingMixIn:
#     def __init__(self, server):
#         self.port = 21
#         self.server = server

#     def allows(self, request, client_address):
#         return threading.alows(client_address, request)
#     def process_request(self, request, client_address):
#         """Start a new thread to process the request."""
#         t = threading.Thread(target = self.do_work, args = (request, client_address))
#         t.daemon = self.daemon_threads
#         t.start()

# class ThreadingTCPServer(ThreadingMixIn, TCPServer):
#     pass

SOURCE_CODE = ['''import threading\n\nclass TCPServer(object):\n    def __init__(self, server):\n        self.port = 22\n        self.server = server\n\n    def process_request(self, request, client_address):\n        self.do_work(request, client_address)\n        self.shutdown_request(request)\n\n    def allows(self, request, client_address):\n        return threading.alows(client_address, request)\n\n    def server_allows(self, request):\n        return threading.alows(self.server, request)\n\nclass ThreadingMixIn:\n    def __init__(self, server):\n        self.port = 21\n        self.server = server\n\n    def allows(self, request, client_address):\n        return threading.alows(client_address, request)\n    def process_request(self, request, client_address):\n        """Start a new thread to process the request."""\n        t = threading.Thread(target = self.do_work, args = (request, client_address))\n        t.daemon = self.daemon_threads\n        t.start()\n\nclass ThreadingTCPServer(ThreadingMixIn, TCPServer):\n    pass''']
SPAN = [Span(30, 0, 30, 52)]
MESSAGE = ["Base classes have conflicting values for attribute 'allows': [[""Function allows""|""relative:///py_file_3382.py:98:5:98:60""]] and [[""Function allows""|""relative:///py_file_3382.py:74:5:74:60""]].\nBase classes have conflicting values for Property 'server': [[""Property server""|""relative:///py_file_3382.py:108:5:108:60""]] and [[""Property server""|""relative:///py_file_3382.py:87:5:87:60""]]."]


class TestDistributableQueryContext(unittest.TestCase):
    desired_block = [[Block(30,
                            31,
                            [30, 31],
                            'class ThreadingTCPServer(ThreadingMixIn, TCPServer):\n    pass',
                            'root.ThreadingTCPServer',
                            CLASS_OTHER,
                            True,
                            'module',
                            ('__', '__class__', 'ThreadingMixIn', 'TCPServer')),
                      Block(3,
                            5,
                            [],
                            'def __init__(self, server):\n        self.port = 22\n        self.server = server',
                            'root.TCPServer.__init__',
                            CLASS_FUNCTION,
                            True,
                            'class TCPServer(object):',
                            ('__', '__class__', 'TCPServer')),
                      Block(7,
                            9,
                            [],
                            'def process_request(self, request, client_address):\n        self.do_work(request, client_address)\n        self.shutdown_request(request)',
                            'root.TCPServer.process_request',
                            CLASS_FUNCTION,
                            False,
                            'class TCPServer(object):',
                            ('__', '__class__', 'TCPServer')),
                      Block(11,
                            12,
                            [],
                            'def allows(self, request, client_address):\n        return threading.alows(client_address, request)',
                            'root.TCPServer.allows',
                            CLASS_FUNCTION,
                            True,
                            'class TCPServer(object):',
                            ('__', '__class__', 'TCPServer')),
                      Block(14,
                            15,
                            [],
                            'def server_allows(self, request):\n        return threading.alows(self.server, request)',
                            'root.TCPServer.server_allows',
                            CLASS_FUNCTION,
                            False,
                            'class TCPServer(object):',
                            ('__', '__class__', 'TCPServer')),
                      Block(18,
                            20,
                            [],
                            'def __init__(self, server):\n        self.port = 21\n        self.server = server',
                            'root.ThreadingMixIn.__init__',
                            'CLASS_FUNCTION',
                            True,
                            'class ThreadingMixIn:',
                            ('__', '__class__', 'ThreadingMixIn')),
                      Block(22,
                            23,
                            [],
                            'def allows(self, request, client_address):\n        return threading.alows(client_address, request)',
                            'root.ThreadingMixIn.allows',
                            CLASS_FUNCTION,
                            True,
                            'class ThreadingMixIn:',
                            ('__', '__class__', 'ThreadingMixIn')),
                      Block(24,
                            28,
                            [],
                            'def process_request(self, request, client_address):\n        """Start a new thread to process the request."""\n        t = threading.Thread(target = self.do_work, args = (request, client_address))\n        t.daemon = self.daemon_threads\n        t.start()',
                            'root.ThreadingMixIn.process_request',
                            CLASS_FUNCTION,
                            False,
                            'class ThreadingMixIn:',
                            ('__', '__class__', 'ThreadingMixIn')),
                      Block(2,
                            15,
                            [2, 6, 10, 13],
                            'class TCPServer(object):\n\n\n',
                            'root.TCPServer',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__', 'object')),
                      Block(17,
                            28,
                            [17, 21],
                            'class ThreadingMixIn:\n',
                            'root.ThreadingMixIn',
                            CLASS_OTHER,
                            False,
                            'module',
                            ('__', '__class__'))]]

    def test_relevant_block(self):
        for i, code in enumerate(SOURCE_CODE):
            span = SPAN[i]
            msg = MESSAGE[i]
            generated_block = get_span_context('Conflicting attributes in base classes',
                                               code, tree_sitter_parser, '',
                                               msg, span, None)

            for j, gen_block in enumerate(generated_block):
                self.assertEqual(self.desired_block[i][j], gen_block)


if __name__ == "__main__":
    unittest.main()
