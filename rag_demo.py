#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 0 最小可跑版 RAG：解析 -> 切分 -> 向量化 -> 检索 -> 生成
依赖: pypdf, requests, numpy
模型: 本地 Ollama（bge-m3 做向量，qwen2.5:1.5b 做生成）
用法:
  python rag_demo.py --pdf "D:/path/book.pdf"
  python rag_demo.py --pdf "D:/path/book.pdf" --rebuild   # 强制重建向量缓存
"""
import argparse
import os
import numpy as np
import requests
from pypdf import PdfReader

OLLAMA = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
GEN_MODEL = "qwen2.5:1.5b"


def ollama_embed(texts):
    """调用 Ollama /api/embeddings 把文本逐条变成向量。"""
    vecs = []
    for t in texts:
        resp = requests.post(OLLAMA + "/api/embeddings",
                             json={"model": EMBED_MODEL, "prompt": t}, timeout=180)
        resp.raise_for_status()
        vecs.append(resp.json()["embedding"])
    return np.array(vecs, dtype=np.float32)


def ollama_generate(prompt):
    resp = requests.post(OLLAMA + "/api/generate",
                         json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
                         timeout=600)
    resp.raise_for_status()
    return resp.json()["response"]


def load_pdf_text(path):
    reader = PdfReader(path)
    return ("\n").join((page.extract_text() or "") for page in reader.pages)


def split_text(text, chunk_size=500, overlap=80):
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
    return chunks


def cache_file(pdf_path, chunk_size, overlap):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    folder = os.path.dirname(os.path.abspath(pdf_path))
    return os.path.join(folder, ".rag_cache_%s_%d_%d.npz" % (base, chunk_size, overlap))


def build_or_load(pdf_path, chunk_size, overlap, rebuild):
    cp = cache_file(pdf_path, chunk_size, overlap)
    if (not rebuild) and os.path.exists(cp):
        d = np.load(cp, allow_pickle=True)
        print("      使用缓存: %s (%d 段)" % (os.path.basename(cp), len(d["chunks"])))
        return list(d["chunks"]), d["vecs"]
    print("      解析 PDF ...")
    text = load_pdf_text(pdf_path)
    print("      共 %d 字符" % len(text))
    print("      切分 ...")
    chunks = split_text(text, chunk_size, overlap)
    print("      切出 %d 段" % len(chunks))
    print("      向量化（Ollama %s，需要一点时间）..." % EMBED_MODEL)
    vecs = ollama_embed(chunks)
    np.savez(cp, chunks=np.array(chunks, dtype=object), vecs=vecs)
    print("      已缓存到 %s" % os.path.basename(cp))
    return chunks, vecs


def retrieve(query, chunks, chunk_vecs, top_k=3):
    qv = ollama_embed([query])[0]
    norms = np.linalg.norm(chunk_vecs, axis=1) * (np.linalg.norm(qv) + 1e-8)
    sims = (chunk_vecs @ qv) / (norms + 1e-8)
    idx = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i]), chunks[i]) for i in idx]


def main():
    parser = argparse.ArgumentParser(description="阶段0 最小 RAG 演示")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量缓存")
    args = parser.parse_args()

    chunks, vecs = build_or_load(args.pdf, args.chunk_size, args.overlap, args.rebuild)
    print("      向量形状: %s" % (vecs.shape,))
    print("")
    print("可以开始提问了（直接回车退出）")

    while True:
        try:
            q = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print("[检索] ...")
        hits = retrieve(q, chunks, vecs, args.top_k)
        for rank, (i, score, chunk) in enumerate(hits, 1):
            print("  [%d] 相似度 %.3f | 第 %d 段" % (rank, score, i))
            print("      %s..." % chunk[:120].replace("\n", " "))
        print("[生成] %s 回答中 ..." % GEN_MODEL)
        context = ("\n\n").join(c for _, _, c in hits)
        prompt = ("你是一个严谨的文档问答助手。请只根据下面的资料回答问题，"
                  "资料里没有的信息就明确说「资料中没有提到」。\n\n"
                  "【资料】\n" + context + "\n\n"
                  "【问题】\n" + q + "\n\n"
                  "【回答】")
        print("")
        print("回答：")
        print(ollama_generate(prompt))
        print("")


if __name__ == "__main__":
    main()
