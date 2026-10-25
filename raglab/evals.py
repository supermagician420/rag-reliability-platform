"""A small, inspectable evaluation harness for grounded QA."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class EvalCase:
    id: str
    question: str
    should_answer: bool
    expected_titles: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    category: str = "answerable"


@dataclass
class EvalRow:
    id: str
    question: str
    should_answer: bool
    status: str
    passed: bool
    retrieval_recall: float
    citation_precision: float
    keyword_coverage: float
    citations: List[str]
    reason: str


def load_cases(path: str) -> List[EvalCase]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("评测文件必须是非空 JSON 数组")
    cases: List[EvalCase] = []
    seen = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("第 %d 条评测用例不是对象" % (index + 1))
        case_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError("评测用例 id 缺失或重复：%s" % case_id)
        if not question:
            raise ValueError("评测用例 %s 缺少 question" % case_id)
        if not isinstance(item.get("should_answer", True), bool):
            raise ValueError("评测用例 %s 的 should_answer 必须是布尔值" % case_id)
        seen.add(case_id)
        cases.append(
            EvalCase(
                id=case_id,
                question=question,
                should_answer=bool(item.get("should_answer", True)),
                expected_titles=[str(value) for value in item.get("expected_titles", [])],
                expected_keywords=[str(value) for value in item.get("expected_keywords", [])],
                category=str(item.get("category", "answerable")),
            )
        )
    return cases


def _recall(expected_titles: Iterable[str], sources: List[Dict[str, Any]]) -> float:
    expected = [title for title in expected_titles if title]
    if not expected:
        return 0.0
    got = {str(source.get("title", "")) for source in sources}
    return len(set(expected) & got) / float(len(set(expected)))


def _keyword_coverage(keywords: Iterable[str], answer: str) -> float:
    expected = [keyword for keyword in keywords if keyword]
    if not expected:
        return 1.0
    lowered = (answer or "").casefold()
    return sum(1 for keyword in expected if keyword.casefold() in lowered) / float(len(expected))


def _citation_precision(result: Any, sources: List[Dict[str, Any]]) -> float:
    if result.status != "answered" or not result.citations:
        return 0.0
    valid = {"[S%d]" % index for index in range(1, len(sources) + 1)}
    return sum(1 for citation in result.citations if citation in valid) / float(len(result.citations))


def run_case(answerer: Any, case: EvalCase, top_k: int = 5) -> EvalRow:
    result = answerer.answer(case.question, top_k=top_k)
    recall = _recall(case.expected_titles, result.sources)
    precision = _citation_precision(result, result.sources)
    coverage = _keyword_coverage(case.expected_keywords, result.answer)

    if case.should_answer:
        passed = (
            result.status == "answered"
            and recall > 0.0
            and coverage >= 0.6
            and len(result.citations) > 0
        )
    else:
        passed = result.status == "refused"

    return EvalRow(
        id=case.id,
        question=case.question,
        should_answer=case.should_answer,
        status=result.status,
        passed=passed,
        retrieval_recall=round(recall, 4),
        citation_precision=round(precision, 4),
        keyword_coverage=round(coverage, 4),
        citations=list(result.citations),
        reason=result.reason,
    )


def evaluate(answerer: Any, cases: List[EvalCase], top_k: int = 5) -> Dict[str, Any]:
    rows = [run_case(answerer, case, top_k=top_k) for case in cases]
    answerable = [row for row in rows if row.should_answer]
    unanswerable = [row for row in rows if not row.should_answer]

    def average(values: List[float]) -> float:
        return round(sum(values) / float(len(values)), 4) if values else 0.0

    false_answers = sum(1 for row in unanswerable if row.status == "answered")
    correct_abstentions = sum(1 for row in unanswerable if row.status == "refused")
    report = {
        "summary": {
            "questions": len(rows),
            "passed": sum(1 for row in rows if row.passed),
            "pass_rate": average([1.0 if row.passed else 0.0 for row in rows]),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "retrieval_recall": average([row.retrieval_recall for row in answerable]),
            "citation_precision": average([row.citation_precision for row in answerable]),
            "keyword_coverage": average([row.keyword_coverage for row in answerable]),
            "abstention_accuracy": average(
                [1.0 if row.status == "refused" else 0.0 for row in unanswerable]
            ),
            "false_answer_rate": round(false_answers / float(len(unanswerable)), 4) if unanswerable else 0.0,
            "correct_abstentions": correct_abstentions,
        },
        "cases": [asdict(row) for row in rows],
    }
    return report


def format_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "RAG reliability evaluation",
        "=" * 62,
        "Questions:            %d" % summary["questions"],
        "Passed:               %d (%.0f%%)" % (summary["passed"], summary["pass_rate"] * 100),
        "Retrieval recall:     %.1f%%" % (summary["retrieval_recall"] * 100),
        "Citation precision:   %.1f%%" % (summary["citation_precision"] * 100),
        "Keyword coverage:     %.1f%%" % (summary["keyword_coverage"] * 100),
        "Abstention accuracy:  %.1f%%" % (summary["abstention_accuracy"] * 100),
        "False answer rate:    %.1f%%" % (summary["false_answer_rate"] * 100),
        "",
        "%-12s %-8s %-9s %-6s %s" % ("id", "expected", "actual", "pass", "reason"),
        "-" * 92,
    ]
    for row in report["cases"]:
        lines.append(
            "%-12s %-8s %-9s %-6s %s"
            % (
                row["id"][:12],
                "answer" if row["should_answer"] else "refuse",
                row["status"],
                "yes" if row["passed"] else "NO",
                row["reason"][:44],
            )
        )
    return "\n".join(lines)


def write_report(report: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
