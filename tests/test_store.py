import os
import tempfile
import unittest

from raglab.seed import seed_store
from raglab.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "rag.db")
        self.store = Store(self.db_path)
        seed_store(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_seed_counts(self):
        self.assertEqual(self.store.count_documents(), 5)
        self.assertGreaterEqual(self.store.count_chunks(), 5)

    def test_chinese_search_finds_leave_policy(self):
        results = self.store.search("病假需要提前几天申请？")
        self.assertTrue(results)
        self.assertEqual(results[0]["title"], "请假与休假")
        self.assertGreaterEqual(results[0]["coverage"], 0.4)

    def test_search_without_evidence_returns_empty(self):
        self.assertEqual(self.store.search("股票期权"), [])

    def test_add_document_and_list(self):
        created = self.store.add_document("测试制度", "这是一段测试内容。第二句。")
        self.assertGreaterEqual(created["chunks"], 1)
        titles = [item["title"] for item in self.store.list_documents()]
        self.assertIn("测试制度", titles)

    def test_empty_document_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_document("空文档", "   ")

    def test_duplicate_documents_are_allowed(self):
        before = self.store.count_documents()
        self.store.add_document("重复", "同样的内容")
        self.store.add_document("重复", "同样的内容")
        self.assertEqual(self.store.count_documents(), before + 2)

    def test_stats(self):
        stats = self.store.stats()
        self.assertEqual(stats["documents"], 5)
        self.assertEqual(stats["chunks"], self.store.count_chunks())


if __name__ == "__main__":
    unittest.main()
