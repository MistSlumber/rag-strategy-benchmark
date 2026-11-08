# rag-strategy-benchmark

在同一份语料、同一份评测集上，对照四种检索策略，**只改变排序逻辑**，看分数到底怎么动。

这个仓库回答的是学 RAG 时最容易被糊弄过去的问题：「加了混合检索、加了重排，就一定会变好吗？」答案是：**不一定，而且我这次跑出来是负结果。**

## 快速开始

```bash
python scripts/run_benchmark.py
python -m unittest discover -s tests -v
```

零第三方依赖、不需要 API key、不需要联网。结果与 `chinese-rag-eval-starter` 共享同一套 20 张自写知识卡和 20 条评测问题，但代码完全自包含，可独立 clone。

## 四种策略

| 策略 | 做法 |
| --- | --- |
| `bm25` | 词面检索基线，中文按单字 + 相邻双字建模 |
| `tfidf` | TF-IDF 向量 + 余弦相似，作为「稠密检索」的零依赖近似 |
| `rrf_hybrid` | 前两路各取 top-10，用 Reciprocal Rank Fusion 融合 |
| `rrf_hybrid_rerank` | 在融合候选上按标题/正文词面覆盖率做启发式重排 |

统一入口是 `src.strategies.rank(strategy, query, indexes, sections, k)`，四种策略共享 `tokenize` 与语料，差异只来自排序。

## 实测结果（k=3，20 题，可复现）

| 策略 | Recall@3 | MRR | 命中 / 总题 | 未命中 |
| --- | --- | --- | --- | --- |
| `bm25` | 0.8500 | 0.7917 | 17 / 20 | 3 |
| `tfidf` | 0.8500 | 0.7417 | 17 / 20 | 3 |
| `rrf_hybrid` | 0.8500 | 0.7917 | 17 / 20 | 3 |
| `rrf_hybrid_rerank` | **0.7000** | **0.5833** | 14 / 20 | 6 |

机器可读取的完整逐条结果在 `results/benchmark.json`。

### 怎么读这个负结果

- **RRF 零增益**：两路召回用的是同一套词面特征（单字 + 双字），只是打分公式不同，排名高度重合。融合两路高度相关的结果，本来就不该期待提升。**混合检索的前提是两路互补**，例如 BM25 + 语义向量，而不是两个词面模型互融。
- **启发式重排掉 0.15**：标题覆盖率权重设得高，遇到口语化提问（「怎么衡量」「瞎编」「费用」）时，把词面碰巧重合但主题无关的片段顶到了前面。这说明**词面特征不足以做重排**——真正的 reranker 需要交叉编码器或语义模型，它看的不是字符重合。
- **三条未命中都是同义鸿沟**：`怎么衡量我的检索系统到底好不好 → Q5 评测集`、`资料里没写的问题模型瞎编 → Q7 幻觉兜底`、`怎么防止费用越用越高 → Q15 成本控制`。词面检索从原理上就抓不到这些。

所以这份实验不是「重排没用」的结论，而是**「在没有语义信号之前，堆叠词面策略只会互相干扰」**的证据。下一步该做的是接一个真正的 embedding 模型或 cross-encoder，再用同一份评测集重跑，而不是继续调这几个权重。

> 声明：`coverage_rerank` 是启发式打分，**不是**交叉编码器重排模型。保留这个变体，是为了把「看起来加了模块、实际变差了」这件事留在仓库里当反面教材。

## 变量控制

| 项 | 是否固定 |
| --- | --- |
| 语料（20 张知识卡） | 固定 |
| 评测问题与 gold 标注 | 固定，只增不改 |
| tokenize / 分块 | 固定 |
| top-k | 固定为 3（可 `--k` 覆盖，但历史分数以 3 为准） |
| 排序逻辑 | **唯一变量** |

自定义实验：`python scripts/run_benchmark.py --k 5 --data data/my_faq.md --questions eval/my_questions.jsonl`。

## 目录

```
src/strategies.py        # 四种策略 + RRF + 启发式重排
src/dataset.py           # 自包含的数据加载
scripts/run_benchmark.py # 生成对照表与 results/benchmark.json
eval/questions.jsonl     # 固定评测集（20 条）
data/faq.md              # 自写示例语料（20 张知识卡）
docs/EXPERIMENT.md       # 实验设计、归因、局限
tests/                   # 9 项回归测试
.github/workflows/ci.yml # Python 3.9 / 3.11 / 3.13 自动跑测试
```

## 语料与许可

`data/faq.md` 与评测集均为自写示例内容。代码 MIT，见 `LICENSE`。评测集是本项目可复现性的锚点，改动措辞会让历史分数失去意义，详见 `CONTRIBUTING.md`。
