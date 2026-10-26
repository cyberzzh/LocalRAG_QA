#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比实验：不同 chunk_size / top_k 下的检索命中率（关键词命中作为代理指标）。"""
import os
import rag

PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "neural_network",
                   "深度学习入门：基于Python的理论与实现.pdf")

QUESTIONS = [
    {"q": "什么是反向传播", "keywords": ["反向传播", "链式法则", "计算图"]},
    {"q": "损失函数有哪些", "keywords": ["损失函数", "均方误差", "交叉熵"]},
    {"q": "什么是梯度下降", "keywords": ["梯度下降", "学习率", "参数更新"]},
    {"q": "感知机是什么", "keywords": ["感知机", "线性", "多层"]},
    {"q": "softmax 函数是什么", "keywords": ["softmax", "概率", "指数"]},
]


def main():
    print("加载文档:", PDF)
    text = rag.load_document(PDF)
    print("字符数:", len(text))
    results = []
    for cs in [256, 512, 1024]:
        chunks = rag.split_text(text, cs, 80)
        print("chunk_size=%d -> %d 段，向量化中..." % (cs, len(chunks)))
        vecs = rag.ollama_embed(chunks)
        for k in [3, 5, 10]:
            n_hit = 0
            for item in QUESTIONS:
                hits = rag.retrieve(item["q"], chunks, vecs, k)
                joined = " ".join(t for _, _, t in hits)
                if any(kw in joined for kw in item["keywords"]):
                    n_hit += 1
            rate = n_hit / len(QUESTIONS)
            results.append((cs, k, rate))
            print("  top_k=%d 命中率=%d/%d=%.0f%%" % (k, n_hit, len(QUESTIONS), rate * 100))
    print("")
    print("| chunk_size | top_k | 命中率 |")
    print("|---|---|---|")
    for cs, k, rate in results:
        print("| %d | %d | %.0f%% |" % (cs, k, rate * 100))


if __name__ == "__main__":
    main()
