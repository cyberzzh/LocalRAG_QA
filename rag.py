#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG 核心模块：解析、切分、向量化、检索、生成。"""
import os
import numpy as np
import requests
from pypdf import PdfReader

OLLAMA = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
GEN_MODEL = "qwen2.5:1.5b"


def load_pdf_text(path):
    reader = PdfReader(path)
    return ("\n").join((page.extract_text() or "") for page in reader.pages)


def load_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path):
    if path.lower().endswith(".pdf"):
        return load_pdf_text(path)
    return load_text_file(path)


def split_text(text, chunk_size=500, overlap=80):
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
    return chunks


def ollama_embed(texts):
    vecs = []
    for t in texts:
        r = requests.post(OLLAMA + "/api/embeddings",
                          json={"model": EMBED_MODEL, "prompt": t}, timeout=180)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return np.array(vecs, dtype=np.float32)


def ollama_generate(prompt):
    r = requests.post(OLLAMA + "/api/generate",
                      json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
                      timeout=600)
    r.raise_for_status()
    return r.json()["response"]


def retrieve(query, chunks, vecs, top_k=3):
    qv = ollama_embed([query])[0]
    norms = np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qv) + 1e-8)
    sims = (vecs @ qv) / (norms + 1e-8)
    idx = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i]), chunks[i]) for i in idx]


def build_prompt(query, hits):
    context = ("\n\n").join(c for _, _, c in hits)
    return ("你是一个严谨的文档问答助手。请只根据下面的资料回答问题，"
            "资料里没有的信息就明确说「资料中没有提到」。\n\n"
            "【资料】\n" + context + "\n\n"
            "【问题】\n" + query + "\n\n"
            "【回答】")


def answer(query, chunks, vecs, top_k=3):
    hits = retrieve(query, chunks, vecs, top_k)
    return ollama_generate(build_prompt(query, hits))
