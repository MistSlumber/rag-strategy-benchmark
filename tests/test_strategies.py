# -*- coding: utf-8 -*-
"""策略实现的回归测试，保证对照实验的四个变体行为稳定。"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import load_questions, load_sections  # noqa: E402
from src.strategies import (BM25Index, STRATEGY_NAMES, TfidfIndex,  # noqa: E402
                            coverage_rerank, rank, rrf_fuse, tokenize)


class TestIndexes(unittest.TestCase):
    def setUp(self):
        self.docs = ["苹果 香蕉 水果", "显卡 显存 训练", "RAG 检索 分块"]

    def test_bm25(self):
        self.assertEqual(BM25Index(self.docs).search("显存不够", k=1), [1])

    def test_tfidf(self):
        self.assertEqual(TfidfIndex(self.docs).search("检索 分块", k=1), [2])

    def test_empty(self):
        self.assertEqual(BM25Index([]).search("任意", k=3), [])
        self.assertEqual(TfidfIndex([]).search("任意", k=3), [])

    def test_tokenize(self):
        self.assertIn("检索", tokenize("RAG 检索"))


class TestFusionAndRerank(unittest.TestCase):
    def test_rrf_prefers_consensus(self):
        merged = rrf_fuse([[1, 2, 3], [1, 3, 2]], top_k=1)
        self.assertEqual(merged, [1])

    def test_rerank_prefers_title_match(self):
        sections = [
            {"id": "Q1", "title": "无关主题", "text": "成本控制 成本 费用"},
            {"id": "Q2", "title": "成本控制", "text": "内容简短"},
        ]
        self.assertEqual(coverage_rerank("成本控制", sections, [0, 1], top_k=1), [1])

    def test_all_strategies_return_valid_ids(self):
        sections = load_sections(os.path.join(ROOT, "data", "faq.md"))
        docs = [s["title"] + "\n" + s["text"] for s in sections]
        indexes = {"bm25": BM25Index(docs), "tfidf": TfidfIndex(docs)}
        for name in STRATEGY_NAMES:
            ids = rank(name, "怎么衡量检索系统", indexes, sections, k=3)
            self.assertIsInstance(ids, list)
            self.assertLessEqual(len(ids), 3)
            self.assertTrue(all(0 <= i < len(sections) for i in ids))

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            rank("nope", "q", {}, [], k=3)


class TestDataset(unittest.TestCase):
    def test_gold_ids_exist(self):
        sections = load_sections(os.path.join(ROOT, "data", "faq.md"))
        questions = load_questions(os.path.join(ROOT, "eval", "questions.jsonl"))
        self.assertEqual(len(sections), 20)
        self.assertEqual(len(questions), 20)
        ids = {s["id"] for s in sections}
        for item in questions:
            self.assertIn(item["gold"], ids)


if __name__ == "__main__":
    unittest.main()
