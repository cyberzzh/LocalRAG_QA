#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradio 网页界面：上传文档 -> 建立索引 -> 问答。"""
import os
import gradio as gr
import rag

STATE = {"chunks": [], "vecs": None, "source": ""}


def index_file(file_path, chunk_size, overlap):
    if not file_path:
        return "请先选择一个 PDF 或 txt 文件"
    text = rag.load_document(file_path)
    chunks = rag.split_text(text, int(chunk_size), int(overlap))
    vecs = rag.ollama_embed(chunks)
    STATE["chunks"] = chunks
    STATE["vecs"] = vecs
    STATE["source"] = os.path.basename(file_path)
    return "已索引《%s》，共 %d 段" % (STATE["source"], len(chunks))


def respond(message, chat_history):
    if not STATE["chunks"]:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "请先上传并建立索引。"})
        return "", chat_history
    ans = rag.answer(message, STATE["chunks"], STATE["vecs"])
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ans})
    return "", chat_history


with gr.Blocks(title="本地 RAG 文档问答助手") as demo:
    gr.Markdown("# 本地 RAG 文档问答助手\n上传文档 → 建立索引 → 提问，答案只基于你的文档。")
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="上传 PDF / txt", type="filepath")
            cs = gr.Slider(128, 1024, value=500, step=64, label="chunk_size")
            ov = gr.Slider(0, 200, value=80, step=10, label="overlap")
            btn = gr.Button("建立索引")
            status = gr.Markdown("等待上传...")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="问答", height=480)
            msg = gr.Textbox(label="你的问题", placeholder="输入问题后回车")
            send = gr.Button("发送")
    btn.click(index_file, [file_in, cs, ov], status)
    send.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
