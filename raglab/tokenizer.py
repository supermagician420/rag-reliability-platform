"""Small, dependency-free tokenizer for mixed Chinese/English text.

SQLite FTS5's unicode61 tokenizer does not segment Chinese reliably, so we
index a token string produced here. Queries use the same tokenizer, which
keeps retrieval deterministic and easy to test.
"""

import re
import unicodedata
from typing import Dict, List


CJK_RANGE = "\u4e00-\u9fff"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[" + CJK_RANGE + r"]+")

STOP_WORDS = set(
    """
    a an and are as at be been being but by can could did do does for from
    had has have he her here him his how i if in into is it its may me might
    more most my no not of on or our out she should so than that the their
    them then there these they this those to too up us was we were what when
    where which who why will with would you your
    """.split()
)

STOP_CHARS = set("的了吗呢啊吧呀是在有和与及对中为请我你他她它们这那个一不也就都而地被把给让从向于以及公司可以需要")

STOP_BIGRAMS = set(
    """
    可以 如何 怎么 怎样 什么 哪些 哪个 是否 需要 请问 麻烦 帮我 告诉 一下
    这个 那个 这些 那些 我们 你们 他们 她们 它们 公司 企业 一个 一些 有关
    关于 对于 以及 还是 或者 而且 但是 因为 所以 如果 已经 正在 可能 应该
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Return unique tokens in first-seen order.

    Chinese runs produce unigrams and bigrams. English/number runs produce
    lower-cased words. Common stop words are removed so a generic word such as
    "company" cannot make an unrelated answer look relevant.
    """

    normalized = unicodedata.normalize("NFKC", text or "").lower()
    ordered: Dict[str, None] = {}
    for match in TOKEN_RE.finditer(normalized):
        piece = match.group(0)
        if piece[0].isascii():
            if piece not in STOP_WORDS and len(piece) > 1:
                ordered[piece] = None
            continue

        # Keep meaningful single characters for short Chinese queries.
        for char in piece:
            if char not in STOP_CHARS:
                ordered[char] = None
        for index in range(len(piece) - 1):
            bigram = piece[index:index + 2]
            if len(bigram) != 2:
                continue
            if bigram in STOP_BIGRAMS:
                continue
            ordered[bigram] = None

    return list(ordered)


def token_text(text: str) -> str:
    """Space-separated token string used by the FTS5 index."""

    return " ".join(tokenize(text))
