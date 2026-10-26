# 本地 RAG 文档问答助手

一个完全本地运行的「上传文档 -> 提问 -> 基于文档回答」的 RAG（检索增强生成）应用。
不依赖任何云端 API，模型和数据都留在本机。

## 功能

- 支持 PDF / txt 文档解析
- 中文语义检索（BGE-M3 embedding + 余弦相似度）
- 基于检索结果的问答（本地 Qwen2.5）
- Gradio 网页界面
- chunk_size / overlap / top_k 可调
- 提供 chunk_size / top_k 对比实验脚本

## 架构

```
用户上传文档
      |
      v
[1] 解析 PDF -> 纯文本          pypdf
      |
      v
[2] 切分 chunk                 按字符 + 重叠
      |
      v
[3] 向量化 embedding           Ollama bge-m3
      |
      v
[4] 检索 top-k                 numpy 余弦相似度
      |
      v
用户提问 -> 问题向量化 -> 检索最相关段落
      |
      v
[5] 拼上下文 + 生成回答         Ollama qwen2.5
```

## 技术栈

- Python 3.12
- Ollama（本地 LLM 运行时）
- bge-m3（中文 embedding）
- qwen2.5:1.5b（生成模型）
- pypdf / numpy / requests
- Gradio（网页界面）

## 快速开始

### 1. 安装 Ollama 并拉模型

```powershell
ollama pull qwen2.5:1.5b
ollama pull bge-m3
```

### 2. 安装 Python 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. 启动网页界面

```powershell
.\.venv\Scripts\python.exe app.py
```

浏览器打开 http://127.0.0.1:7860 即可使用。

### 4. 命令行版

```powershell
.\.venv\Scripts\python.exe rag_demo.py --pdf "你的文件.pdf"
```

### 5. 对比实验

```powershell
.\.venv\Scripts\python.exe experiment.py
```

## 实验结果

运行 experiment.py 后把结果填到这里（下面为示例格式）：

| chunk_size | top_k | 命中率 |
|---|---|---|
| 256 | 3 | - |
| 256 | 5 | - |
| 256 | 10 | - |
| 512 | 3 | - |
| 512 | 5 | - |
| 512 | 10 | - |
| 1024 | 3 | - |
| 1024 | 5 | - |
| 1024 | 10 | - |

## 项目结构

```
rag.py            核心：解析 / 切分 / 向量化 / 检索 / 生成
app.py            Gradio 网页界面
experiment.py     对比实验
rag_demo.py       命令行演示
requirements.txt  依赖
```

## 设计说明

- 为什么用 BGE-M3：中文语义检索效果好的开源 embedding 模型，支持多语言。
- 为什么 chunk_size 可调：不同文档适合的切分粒度不同，需要通过实验确定。
- 为什么本地部署：数据隐私 + 零 API 成本，文档不出本机。
- 为什么用余弦相似度：embedding 已归一化，余弦相似度等价于内积，计算简单且效果稳定。
