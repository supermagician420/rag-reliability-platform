import unittest

from raglab.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   "), [])

    def test_rejects_too_small_limit(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", max_chars=20)

    def test_sentence_aware_chunks_respect_limit(self):
        text = "第一句话。" * 80
        chunks = chunk_text(text, max_chars=100)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_paragraphs_and_bullets(self):
        chunks = chunk_text("标题\n\n- 第一项\n- 第二项\n\n结尾。", max_chars=200)
        self.assertEqual(len(chunks), 1)
        self.assertIn("第一项", chunks[0])
        self.assertIn("结尾", chunks[0])


if __name__ == "__main__":
    unittest.main()
