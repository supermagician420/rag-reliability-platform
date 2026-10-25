import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from raglab.cli import main


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp_dir.name, "rag.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_and_ask_json(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["init", "--db", self.db]), 0)
            code = main(["ask", "病假需要提前几天申请？", "--db", self.db, "--json"])
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["ask", "病假需要提前几天申请？", "--db", self.db, "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["citations"], ["[S1]"])

    def test_empty_question_exits_nonzero(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["init", "--db", self.db]), 0)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["ask", "", "--db", self.db]), 2)

    def test_ingest_text_file(self):
        path = os.path.join(self.temp_dir.name, "policy.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("这是新制度。")
        with redirect_stdout(io.StringIO()) as output:
            code = main(["ingest", path, "--db", self.db, "--title", "新制度"])
        self.assertEqual(code, 0)
        self.assertIn("新制度", output.getvalue())

    def test_eval_default_suite_passes(self):
        with redirect_stdout(io.StringIO()) as output:
            code = main(["eval", "--db", self.db])
        self.assertEqual(code, 0)
        self.assertIn("Passed:", output.getvalue())
        self.assertIn("100%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
