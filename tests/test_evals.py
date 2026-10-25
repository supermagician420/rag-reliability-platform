import json
import os
import tempfile
import unittest

from raglab.answer import make_mock_answerer
from raglab.evals import EvalCase, evaluate, format_report, load_cases, run_case, write_report
from raglab.seed import seed_store
from raglab.store import Store


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.temp_dir.name, "rag.db"))
        seed_store(self.store)
        self.answerer = make_mock_answerer(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_answerable_case_passes(self):
        case = EvalCase(
            id="leave",
            question="病假需要提前几天申请？",
            should_answer=True,
            expected_titles=["请假与休假"],
            expected_keywords=["1 个工作日", "医院证明"],
        )
        row = run_case(self.answerer, case)
        self.assertTrue(row.passed)
        self.assertEqual(row.status, "answered")

    def test_unanswerable_case_passes_on_refusal(self):
        case = EvalCase(id="stock", question="公司有股票期权吗？", should_answer=False)
        row = run_case(self.answerer, case)
        self.assertTrue(row.passed)
        self.assertEqual(row.status, "refused")

    def test_wrong_expected_title_fails(self):
        case = EvalCase(
            id="wrong",
            question="病假需要提前几天申请？",
            should_answer=True,
            expected_titles=["不存在的文档"],
            expected_keywords=["1 个工作日"],
        )
        self.assertFalse(run_case(self.answerer, case).passed)

    def test_evaluate_aggregates(self):
        cases = [
            EvalCase(id="a", question="病假需要提前几天申请？", should_answer=True,
                     expected_titles=["请假与休假"], expected_keywords=["医院证明"]),
            EvalCase(id="b", question="公司有股票期权吗？", should_answer=False),
        ]
        report = evaluate(self.answerer, cases)
        self.assertEqual(report["summary"]["questions"], 2)
        self.assertEqual(report["summary"]["passed"], 2)
        self.assertEqual(report["summary"]["false_answer_rate"], 0.0)
        self.assertIn("Passed", format_report(report))

    def test_load_cases_rejects_duplicate_ids(self):
        path = os.path.join(self.temp_dir.name, "cases.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([
                {"id": "x", "question": "q1", "should_answer": True},
                {"id": "x", "question": "q2", "should_answer": True},
            ], handle)
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_write_report(self):
        path = os.path.join(self.temp_dir.name, "out", "report.json")
        write_report({"summary": {"passed": 1}}, path)
        with open(path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["summary"]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
