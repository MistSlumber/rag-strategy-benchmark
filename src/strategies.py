# -*- coding: utf-8 -*-
"""四种检索策略的零依赖对照实现。

BM25 / TF-IDF 余弦 / RRF 混合 / 混合+启发式重排，共享同一套 tokenize、
同一份语料和同一份评测集，差异只来自排序逻辑，分数变化因此可以归因。

注意：coverage_rerank 是词面覆盖度的启发式打分，**不是**交叉编码器重排模型。
它的作用是演示「重排这一层能带来什么」，不能当作生产级 reranker 的效果证据。
"""
import math
import re
from collections import Counter, defaultdict

STRATEGY_NAMES = ("bm25", "tfidf", "rrf_hybrid", "rrf_hybrid_rerank")


def tokenize(text: str) -> list:
    text = text.lower()
    units = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    bigrams = [units[i] + units[i + 1] for i in range(len(units) - 1)]
    return units + bigrams


class BM25Index:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in self.docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        self.avg = self.avg or 1.0
        df = Counter()
        for tf in self.tfs:
            for term in tf:
                df[term] += 1
        total = len(self.docs)
        self.idf = {t: math.log(1 + (total - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}

    def search(self, query, k=3):
        qtf = Counter(tokenize(query))
        scored = []
        for idx, tf in enumerate(self.tfs):
            length = self.lens[idx] or 1
            score = 0.0
            for term, qfreq in qtf.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avg)
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1) / denom) * qfreq
            scored.append((idx, score))
        scored.sort(key=lambda pair: -pair[1])
        return [idx for idx, score in scored if score > 0][:k]


class TfidfIndex:
    """TF-IDF 向量 + 余弦相似，作为「稠密检索」的零依赖近似对照。"""

    def __init__(self, docs):
        self.docs = list(docs)
        tfs = [Counter(tokenize(d)) for d in self.docs]
        df = Counter()
        for tf in tfs:
            for term in tf:
                df[term] += 1
        total = len(self.docs)
        self.idf = {t: math.log((total + 1) / (f + 1)) + 1.0
                    for t, f in df.items()}
        self.vecs = []
        for tf in tfs:
            vec = {t: (1 + math.log(f)) * self.idf.get(t, 1.0)
                   for t, f in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vecs.append({t: v / norm for t, v in vec.items()})

    def search(self, query, k=3):
        qtf = Counter(tokenize(query))
        qvec = {t: (1 + math.log(f)) * self.idf.get(t, 1.0)
                for t, f in qtf.items()}
        norm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        qvec = {t: v / norm for t, v in qvec.items()}
        scored = []
        for idx, vec in enumerate(self.vecs):
            score = sum(weight * vec.get(term, 0.0)
                        for term, weight in qvec.items())
            scored.append((idx, score))
        scored.sort(key=lambda pair: -pair[1])
        return [idx for idx, score in scored if score > 0][:k]


def rrf_fuse(rankings, k=60, top_k=3):
    """Reciprocal Rank Fusion：按名次倒数累加，不依赖各路分数是否可比。"""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    return [doc_id for doc_id, _ in ordered][:top_k]


def coverage_rerank(query, sections, candidate_ids, top_k=3):
    """启发式重排：标题覆盖率权重高于正文，并对超长片段做轻微惩罚。"""
    qterms = set(tokenize(query))
    if not qterms:
        return list(candidate_ids)[:top_k]
    scored = []
    for idx in candidate_ids:
        title_terms = set(tokenize(sections[idx]["title"]))
        body_terms = set(tokenize(sections[idx]["text"]))
        title_cov = len(qterms & title_terms) / len(qterms)
        body_cov = len(qterms & body_terms) / len(qterms)
        length_penalty = min(len(sections[idx]["text"]), 800) / 800 * 0.05
        scored.append((idx, 2.0 * title_cov + body_cov - length_penalty))
    scored.sort(key=lambda pair: -pair[1])
    return [idx for idx, _ in scored[:top_k]]


def rank(strategy, query, indexes, sections, k=3):
    """统一入口，返回文档下标列表。"""
    if strategy == "bm25":
        return indexes["bm25"].search(query, k=k)
    if strategy == "tfidf":
        return indexes["tfidf"].search(query, k=k)
    if strategy in ("rrf_hybrid", "rrf_hybrid_rerank"):
        pool = [
            indexes["bm25"].search(query, k=10),
            indexes["tfidf"].search(query, k=10),
        ]
        fused = rrf_fuse(pool, top_k=10)
        if strategy == "rrf_hybrid":
            return fused[:k]
        return coverage_rerank(query, sections, fused, top_k=k)
    raise ValueError("未知策略: %s" % strategy)
