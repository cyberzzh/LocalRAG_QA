# 阶段 0:本地 RAG 文档问答助手 —— 详细计划

> 这是你简历的第一个完整项目。目标不是"用最牛的模型",而是**把 RAG 的每个环节亲手跑通、能讲清楚、能展示**。

---

## 一、这个项目最终长什么样(先想清楚终点)

一个网页应用,流程是:

```
上传文档(如《深度学习入门》PDF)
        ↓
[1] 解析 → 抽成纯文本
        ↓
[2] 切分 → 切成一段段 chunk
        ↓
[3] 向量化 → 每段用 embedding 模型变成向量
        ↓
[4] 存入向量库(Chroma)
──────────────────────────────
你输入问题:"反向传播的链式法则是什么?"
        ↓
[5] 问题也向量化 → 在库里找最相似的 top_k 段
        ↓
[6] (可选)重排 → rerank 把最相关的排前面
        ↓
[7] 拼接上下文 + 问题 → 喂给 LLM(Qwen)
        ↓
返回:基于文档的回答 + 引用来源(来自哪一页/哪段)
```

**完成标准(Definition of Done):**
- [ ] 能上传至少 PDF 和 txt 两种格式
- [ ] 能回答文档里有的问题,且**带引用来源**(哪一页/哪一段)
- [ ] 有网页界面(Gradio),能录屏演示
- [ ] README 写好 + 30 秒录屏

---

## 二、环境准备(Windows)

### 1. Python 环境
- 建议 Python 3.10 或 3.11(3.12+ 个别依赖可能还不稳)。
- 建虚拟环境(避免污染全局):
  ```powershell
  cd D:\universitystudy\accumulate\阶段0_RAG文档问答助手
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

### 2. Ollama(本地跑 LLM)
- 去 https://ollama.com 下载 Windows 安装包,装完会自动起服务。
- 拉模型(按显卡选,别贪大):
  - 有 NVIDIA 显卡(≥8G 显存):`ollama pull qwen2.5:7b`
  - 无显卡 / 显存小:`ollama pull qwen2.5:3b`(甚至 `qwen2.5:1.5b`)
- 验证:`ollama run qwen2.5:3b` 能对话即可。

### 3. 核心依赖
```powershell
pip install llama-index-core llama-index-embeddings-huggingface llama-index-vector-stores-chroma
pip install sentence-transformers gradio chromadb pypdf
```
> 用 LlamaIndex 的原因:RAG 场景它对新手最友好,`from_documents` 一条链就能跑通,之后每步再拆开看。

---

## 三、分 7 步实现(按顺序,每一步单独可运行、可验证)

### Step 1 —— 文档解析
把 PDF/txt 转成纯文本。LlamaIndex 用 `SimpleDirectoryReader` 一行搞定,内部对 PDF 走 pypdf。
- 验证点:打印解析出的前 500 字,确认没有乱码(中文 PDF 尤其要检查)。
- 测试文档:直接用你目录里的《深度学习入门》PDF。

### Step 2 —— 切分 chunk
把长文本切成小块。用 `SentenceSplitter`。
- 核心参数:`chunk_size`(每块 token 数)和 `chunk_overlap`(重叠,一般 10%~20%)。
- 这步**必须调参对比**:256 / 512 / 1024 各跑一次,看哪个检索更准(见"实验"章节)。

### Step 3 —— 向量化 embedding
把每段文字变成向量。**中文用 BGE 系列**:
- 轻量:`BAAI/bge-small-zh-v1.5`(24M 参数,CPU 也能跑)
- 更强:`BAAI/bge-large-zh-v1.5`
- 用 `HuggingFaceEmbedding` 加载,embedding 和后面 LLM 是**两个独立模型**,别搞混。

### Step 4 —— 存库 + 检索
把向量存进 Chroma,再根据问题向量检索最相似的 top_k 段。
- 核心参数:`top_k`(返回几段,3 / 5 / 10 对比)。
- 验证点:先不接 LLM,单纯打印"问题命中了哪些段落",确认检索本身是对的。**这一步对了,后面才对。**

### Step 5 —— 生成回答(接 LLM)
把检索到的段落拼成上下文,用提示词模板喂给 Ollama 的 Qwen。
- 提示词模板要点:明确"只根据给定资料回答,资料没有就说不知道",并要求带来源。
- 用 `Ollama` 接口,开启流式输出(体验更好、面试也能讲)。

### Step 6 —— 界面(Gradio)
用 Gradio Blocks 做一个两栏页面:左边上传文件 + 显示"已索引状态",右边聊天框问答,回答里带来源标注。

### Step 7 —— 优化(选做但强烈推荐)
- 加 **rerank**:检索出 20 段 → 用 `BAAI/bge-reranker-large` 重排取前 3~5 段,准确率通常明显提升。
- 这一步是简历里"对比了不同检索策略"的关键素材。

---

## 四、必做的对比实验(这是简历的硬通货)

用 **3~5 个固定问题**,分别测下面几组配置,记录回答质量和"检索是否命中":

| 实验 | 变量 | 记录什么 |
|---|---|---|
| 1 | chunk_size:256 / 512 / 1024 | 命中率、回答是否漏信息 |
| 2 | top_k:3 / 5 / 10 | 是否把无关段落也塞进去导致跑偏 |
| 3 | 有无 rerank | 命中率前后对比 |

把这几个对比结果做成表格放 README,面试时直接讲——**这就是"对比了不同切分与检索策略"那句话的底气**。

---

## 五、Windows 常见坑(提前避雷)

1. **中文路径/编码**:读文件统一用 `encoding="utf-8"`,遇到 GBK 文件先转。
2. **Ollama 没起服务**:模型调用报连接错误时,先确认 Ollama 托盘图标在跑,或 `ollama serve`。
3. **无显卡跑得慢**:CPU 跑 1.5b/3b 完全能出结果,慢一点正常,别因此卡住。
4. **embedding 模型下载慢**:BGE 模型从 HuggingFace 下,国内可用 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。
5. **显存不够**:LLM 用 Ollama 量化版、embedding 用小模型,两者都别用大参数版。

---

## 六、简历 & README 要突出什么

- **做了什么**:本地文档问答系统,PDF 解析 → 中文语义检索 → 流式生成。
- **难点/决策**:为什么 chunk 选这个大小、为什么加 rerank、embedding 为什么选 BGE 而不是通用的。
- **量化结果**:rerank 让命中率从 X% 提到 Y%(实验表格直接搬过来)。

---

## 七、下一步

确认本机有没有 NVIDIA 显卡、选 LlamaIndex(还是想试 LangChain),我就可以直接帮你把项目脚手架 + 第一版能跑的代码写出来。
