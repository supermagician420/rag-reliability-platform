import unittest

from raglab.tokenizer import token_text, tokenize


class TokenizerTests(unittest.TestCase):
    def test_mixed_chinese_and_english(self):
        tokens = tokenize("IT 报修，电脑坏了！")
        self.assertIn("报修", tokens)
        self.assertIn("电脑", tokens)
        self.assertNotIn("，", tokens)

    def test_question_bigrams_are_kept(self):
        tokens = tokenize("请假需要提前几天申请")
        self.assertIn("请假", tokens)
        self.assertIn("申请", tokens)
        self.assertNotIn("需要", tokens)

    def test_stop_words_are_removed(self):
        self.assertEqual(tokenize("the and are"), [])
        self.assertEqual(tokenize("   "), [])

    def test_token_text_is_space_separated(self):
        value = token_text("病假 1 天")
        self.assertIn("病假", value)
        self.assertNotIn("  ", value)


if __name__ == "__main__":
    unittest.main()
