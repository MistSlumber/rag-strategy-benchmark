# -*- coding: utf-8 -*-
"""在固定评测集上跑四种策略，输出 Markdown 对照表与 results/benchmark.json。"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import load_questions, load_sections  # noqa: E402
from src.strategies import (BM25Index, STRATEGY_NAMES, TfidfIndex,  # noqa: E402
                            rank)


def score_strategy(strategy, indexes, sections, questions, k):
    hit, reciprocal, failures = 0, 0.0, []
    for item in questions:
        ranked = rank(strategy, item["question"], indexes, sections, k=k)
        ids = [sections[idx]["id"] for idx in ranked]
        gold = item["gold"]
        position = ids.index(gold) + 1 if gold in ids else None
        if position:
            hit += 1
            reciprocal += 1.0 / position
        else:
            failures.append({"question": item["question"], "gold": gold, "top_k": ids})
    total = len(questions) or 1
    return {
        "recall_at_k": round(hit / total, 4),
        "mrr": round(reciprocal / total, 4),
        "hits": hit,
        "total": len(questions),
        "failures": failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="检索策略对照实验")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "faq.md"))
    parser.add_argument("--questions", default=os.path.join(ROOT, "eval", "questions.jsonl"))
    args = parser.parse_args(argv)

    sections = load_sections(args.data)
    questions = load_questions(args.questions)
    if not sections or not questions:
        raise SystemExit("语料或评测集为空")
    docs = [s["title"] + "\n" + s["text"] for s in sections]
    indexes = {"bm25": BM25Index(docs), "tfidf": TfidfIndex(docs)}

    results = {"k": args.k, "docs": len(sections), "questions": len(questions),
               "strategies": {}}
    for name in STRATEGY_NAMES:
        results["strategies"][name] = score_strategy(
            name, indexes, sections, questions, args.k)

    out = os.path.join(ROOT, "results", "benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("| 策略 | Recall@%d | MRR | 命中 / 总题 | 未命中 |" % args.k)
    print("| --- | --- | --- | --- | --- |")
    for name in STRATEGY_NAMES:
        item = results["strategies"][name]
        print("| %s | %.4f | %.4f | %d / %d | %d |" % (
            name, item["recall_at_k"], item["mrr"], item["hits"],
            item["total"], len(item["failures"])))
    print("\n结果写入 results/benchmark.json")
    return results


if __name__ == "__main__":
    main()
