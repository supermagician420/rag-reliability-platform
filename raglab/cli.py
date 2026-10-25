"""Command-line entry point for ingestion, Q&A, evaluation, and the web UI."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from .answer import Answerer, make_mock_answerer
from .evals import evaluate, format_report, load_cases, write_report
from .llm import LLMError, OpenAICompatibleLLM
from .seed import seeded_store, seed_store
from .store import Store


DEFAULT_DB = os.path.join("data", "rag.db")
DEFAULT_CASES = os.path.join(os.path.dirname(__file__), "eval_cases.json")


def _llm_from_mode(mode: str) -> Any:
    if mode == "openai":
        return OpenAICompatibleLLM()
    return None


def _public_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": source.get("chunk_id"),
            "title": source.get("title"),
            "snippet": source.get("snippet", ""),
            "score": source.get("score", 0.0),
            "coverage": source.get("coverage", 0.0),
        }
        for source in sources
    ]


def _make_answerer(store: Store, llm_mode: str) -> Answerer:
    llm = _llm_from_mode(llm_mode)
    if llm is None:
        return make_mock_answerer(store)
    return Answerer(store, llm)


def cmd_init(args: argparse.Namespace) -> int:
    store = Store(args.db)
    if store.count_documents() > 0:
        print("数据库已有 %d 篇文档，跳过初始化。" % store.count_documents())
        return 0
    if not args.empty:
        seed_store(store)
    print("已初始化 %s：%d 篇文档，%d 个分块。" % (args.db, store.count_documents(), store.count_chunks()))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        print("文件不存在：%s" % path, file=sys.stderr)
        return 2
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    title = args.title or os.path.splitext(os.path.basename(path))[0]
    store = Store(args.db)
    try:
        result = store.add_document(title, text)
    except ValueError as exc:
        print("导入失败：%s" % exc, file=sys.stderr)
        return 2
    print("已导入《%s》：%d 个分块。" % (result["title"], result["chunks"]))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    store = seeded_store(args.db)
    answerer = _make_answerer(store, args.llm)
    try:
        result = answerer.answer(args.question, top_k=args.top_k)
    except ValueError as exc:
        print("参数错误：%s" % exc, file=sys.stderr)
        return 2
    payload = result.to_dict()
    payload["sources"] = _public_sources(result.sources)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("状态：%s" % result.status)
        print(result.answer)
        if result.citations:
            print("引用：%s" % ", ".join(result.citations))
        if result.reason:
            print("说明：%s" % result.reason)
        if result.sources:
            print("检索：")
            for index, source in enumerate(result.sources, start=1):
                print("  [S%d] %s (score %.3f, coverage %.3f)" % (
                    index, source.get("title", ""), source.get("score", 0.0), source.get("coverage", 0.0)
                ))
    if result.status == "error":
        return 2
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    store = seeded_store(args.db)
    answerer = _make_answerer(store, args.llm)
    cases = load_cases(args.cases)
    report = evaluate(answerer, cases, top_k=args.top_k)
    print(format_report(report))
    if args.report:
        write_report(report, args.report)
        print("\nJSON 报告：%s" % os.path.abspath(args.report))
    return 0 if report["summary"]["passed"] == report["summary"]["questions"] else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.server import create_server

    server = create_server(args.host, args.port, args.db, args.llm)
    print("RAG Reliability Lab: http://%s:%d" % (args.host, server.server_port))
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raglab", description="Zero-dependency grounded RAG reliability lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=DEFAULT_DB, help="SQLite 数据库路径（默认 %s）" % DEFAULT_DB)
    common.add_argument("--llm", choices=["mock", "openai"], default="mock", help="mock 离线演示 / openai 兼容接口")

    init = subparsers.add_parser("init", parents=[common], help="创建数据库并导入示例文档")
    init.add_argument("--empty", action="store_true", help="只建空库，不导入示例文档")
    init.set_defaults(func=cmd_init)

    ingest = subparsers.add_parser("ingest", parents=[common], help="导入一个 UTF-8 文本文件")
    ingest.add_argument("file")
    ingest.add_argument("--title", default="", help="文档标题，默认使用文件名")
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", parents=[common], help="回答一个问题")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    evaluate_parser = subparsers.add_parser("eval", parents=[common], help="运行评测集")
    evaluate_parser.add_argument("--cases", default=DEFAULT_CASES)
    evaluate_parser.add_argument("--top-k", type=int, default=5)
    evaluate_parser.add_argument("--report", default="", help="可选：写出 JSON 报告")
    evaluate_parser.set_defaults(func=cmd_eval)

    serve = subparsers.add_parser("serve", parents=[common], help="启动本地 Web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LLMError as exc:
        print("LLM 配置或调用失败：%s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
