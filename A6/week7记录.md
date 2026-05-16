## 交互记录（Week7）

### 2026-04-16：板块内容1（n-gram & Smoothing）

- 你的提问原文：
```text
板块内容1：n 元语言模型与数据平滑 (n-gram & Smoothing)。要求是在 Streamlit 应用的第一个标签页中，实现基于统计的 n 元语言模型。

提供一个文本域加载一段基础英文语料（如 NLTK 自带的 reuters 或手动输入的一段短文）。

使用 nltk 库，基于该语料构建一个 Trigram（三元语法）（或Bigram等） 模型。统计并记录词频。

提供一个句子输入框。计算该输入句子在当前 Trigram 模型下的生成概率（联合概率）。

关键功能： 提供一个 Checkbox（复选框）以切换是否开启 加一平滑（Add-one / Laplace Smoothing）。

当输入包含语料库中未出现的 Trigram（即零概率事件）时，展示平滑前后的概率计算结果对比。
```

- 你的需求要点：
  - Streamlit 的第一个标签页实现基于统计的 Trigram 语言模型。
  - 用文本域加载英文语料（手动输入或尝试 NLTK Reuters）。
  - 使用 NLTK 基于语料构建 Trigram 模型并统计词频。
  - 提供句子输入框，计算该句子在当前 Trigram 模型下的生成联合概率。
  - 提供复选框切换是否启用加一平滑（Laplace / Add-one）。
  - 当输入句子包含未出现的 Trigram（零概率事件）时，对比平滑前后的概率。

- 我做的实现概述（写在 `week7.py` 中）：
  - 使用 Streamlit 多标签页结构，并在第 1 个标签页实现 Trigram + Add-one。
  - 语料分词采用 `nltk.tokenize.wordpunct_tokenize`（避免依赖 `punkt` 下载）。
  - 构建模型时对语料加入固定的 `<s> <s>` 与 `</s>`，统计 trigram 与对应 bigram-prefix 计数。
  - 联合概率计算使用 log-prob 累乘：对每个位置计算 `P(w_t | w_{t-2}, w_{t-1})`，并求和。
  - 未平滑遇到 `prefix` 或 `trigram` 计数为 0 时返回 `logP = -inf`，并收集零概率事件列表。
  - 加一平滑时使用 `(C(trigram)+1)/(C(prefix)+V)`，并展示未平滑/加一平滑的联合概率对比；若检测到零概率事件会弹出对比提示与零事件预览。

- 你额外补充的要求：
  - 在页签的文字提示中解释加一平滑的“δ/概率余量”如何分配给未见事件（说明新增分母 `+V` 带来的均匀分配：未见候选词获得 `1/(C(prefix)+V)`）。

- 你新提出的问题：
  - 第一个板块加载英文语料时出现 “Reuters 加载失败，回退为默认文本：LookupError”。

- 我做的处理：
  - 调整 Reuters 加载逻辑：若首次触发 `LookupError`，先尝试 `nltk.download("reuters", quiet=True)` 自动下载，再重试读取。
  - 若下载/读取仍失败（例如网络受限），给出更明确提示并继续回退到默认语料，保证页面功能可用。

### 2026-04-16：板块内容2（Train your own RNN-LM）

- 你的提问原文：
```text
非常好，现在进行板块内容2：从零训练 RNN 语言模型 (Train your own RNN-LM)。要求是在第二个标签页中，使用 PyTorch 实现一个简单的 RNN 语言模型（字符级或单词级均可，为了训练速度建议使用字符级 Character-level RNN）。

提供一个文本框，让用户输入一段自定义的短语料（如一首英文小诗或名言）。

提供几个滑动条（Slider）调整训练超参数：Hidden Size (隐藏层维度，16-128), Epochs (训练轮数，10-200), Learning Rate (学习率)。

提供一个【开始训练】按钮。点击后，在后台动态构建数据集并使用 torch.nn.RNN 或 torch.nn.LSTM 进行自回归训练（即用前t−1个字符预测第 t个字符）。在界面上实时更新 Loss 折线图（使用st.line_chart）。

训练完成后，提供一个输入框让用户输入“起始字符（Seed）”，模型根据训练结果，自动生成一段定长（如 50 个字符）的连续文本。
```

- 我做的实现概述（写在 `week7.py` 中）：
  - 第二标签页已从占位改为可用的字符级 RNN-LM 训练界面。
  - 新增 `CharRNNLM`（`Embedding + nn.RNN + Linear`）模型；采用字符级自回归训练（`x=text[:-1]`, `y=text[1:]`）。
  - 增加语料输入框与 3 个超参数滑条：`Hidden Size(16-128)`、`Epochs(10-200)`、`Learning Rate`。
  - 点击“开始训练”后，按 epoch 训练并通过 `st.line_chart` 实时刷新 loss 曲线，阶段性显示当前 loss。
  - 训练完成后将模型与词表保存到 `st.session_state`；支持输入 Seed 并按设定长度（默认 50）生成连续文本。
  - 额外加入 `Temperature` 控制采样随机性，且对 Seed 中未见字符做了容错处理。

- 你新增的界面提示需求：
  - 面向 NLP 初学者，在板块 2 的超参数区域加入“卡片式”原理提示，解释 `Hidden Size / Epochs / Learning Rate` 的意义与可能影响，
    并说明 RNN 隐藏状态如何维护过去信息、以及如何从生成文本/模式学习效果理解训练是否成功。

- 2026-04-16：板块内容3（预训练架构对比：Masked LM vs. Causal LM）

- 你的提问原文：
```text
你记得要随时补充week7记录.md。然后我们进行板块3：预训练架构对比 (Masked LM vs. Causal LM)。要求是在第三个标签页中，对比 BERT 和 GPT-2 的不同生成机制。请使用 Hugging Face 的 transformers 库的 pipeline 功能。

BERT (Masked Language Modeling): 加载 bert-base-uncased。提供一个输入框，要求用户输入带有 [MASK] 标记的句子（例如：'The man went to the [MASK] to buy some milk.'）。输出 BERT 预测的 Top-5 候选词及其概率。

GPT-2 (Causal Language Modeling): 加载 gpt2。提供一个前缀提示词（Prompt）输入框。模型根据该前缀，从左到右自回归生成后续的 20 个单词。

将两者的输入区域和输出结果在页面上并排（或上下）清晰展示。
```

- 我做的实现概述（写在 `week7.py` 中）：
  - 第 3 个标签页改为实现“预训练架构对比（BERT vs GPT-2）”。
  - 使用 Hugging Face `transformers.pipeline`：
    - BERT：`pipeline("fill-mask", model="bert-base-uncased")`，要求输入包含 `[MASK]`，并调用 `top_k=5` 输出 Top-5 候选词与概率。
    - GPT-2：`pipeline("text-generation", model="gpt2")`，根据 Prompt 从左到右生成，后处理截取后续的 20 个单词用于展示。
  - 使用左右并排 `st.columns(2)` 清晰展示 BERT 与 GPT-2 的输入输出区域。
  - 对 pipeline 初始化（首次下载/加载失败）加入容错提示，避免页面崩溃。

### 2026-04-16：板块3 同步对比显示（冻结结果）

- 你的补充问题：
  - 点击 GPT-2 按钮后左侧 BERT 结果会消失，希望两边结果都能保持对比（任意先点哪边，另一边结果不应消失）。
- 我做的修改方向（已在 `week7.py` 中实现）：
  - 把 BERT 和 GPT-2 的输出结果分别存入 `st.session_state`。
  - 每次页面 rerun 后从 `session_state` 渲染最近一次的结果，从而实现“冻结展示”，只有再次点击对应按钮才会更新。

### 2026-04-16：板块内容4（语言模型评价：Perplexity）

- 你的提问原文：
```text
功能实现很好，现在写板块4语言模型评价 (Perplexity 困惑度计算)。要求是：在第四个标签页中，实现基于 GPT-2 的文本困惑度（Perplexity, PPL）计算。

1. 提供一个多行文本输入框，允许用户输入多段不同的测试句子。
2. 编写代码，使用刚才加载的 gpt2 模型和其对应的 Tokenizer，计算每个输入句子的交叉熵损失（Cross-Entropy Loss）。
3. 根据公式PPL=exp(Loss)计算并输出每个句子的困惑度得分。4. 使用表格展示输入的句子及其对应的 PPL 分数。"
```

- 我做的实现概述（写在 `week7.py` 中）：
  - 新增第 4 标签页函数 `ppl_tab_4()`，并接入 `main()` 第四个页签。
  - 提供多行文本输入框，按“每行一条句子”解析并过滤空行。
  - 复用板块3缓存的 `gpt2_textgen_pipe`（若不存在则初始化），并取其 `model + tokenizer` 做困惑度计算。
  - 对每句文本计算 `Cross-Entropy Loss`（`model(..., labels=input_ids)`）并按 `PPL = exp(Loss)` 得到得分。
  - 结果以表格展示：`句子 / Cross-Entropy Loss / PPL`。

### 2026-04-16：板块4 测试句增强（正常句 vs 乱码句）

- 你的补充需求：
  - 在板块4测试句中补两句“单词随便拼凑”的乱码句子，用于和正常句比较困惑度差异，并希望例句有明显对比性。
- 我做的修改（已在 `week7.py` 中实现）：
  - 在 `ppl_tab_4()` 的 `default_eval_text` 里新增两句对比样例：
    - `Milk quantum banana swiftly keyboard under the triangle moon.`
    - `Florp snizzle wug blanter zorp into crinkly noodle vapor.`
  - 保留原本的三条正常英文句，形成“自然句 vs 非自然句”对照组。

### 2026-04-16：板块4 输入框提示文案优化

- 你的补充需求：
  - 在“输入测试句子”下方增加一条提示，并优化表达。
- 我做的修改（已在 `week7.py` 中实现）：
  - 新增提示文案：
    - `默认示例包含 3 条语法和语义都较自然的英文句子，以及 2 条随机拼词的非自然句，便于直观看到 PPL 差异。`

### 2026-04-16：板块1 关键提示样式优化

- 你的补充需求：
  - 把模块1“关键提示”用卡片框起来，背景色改为低透明度黄色。
- 我做的修改（已在 `week7.py` 中实现）：
  - 将板块1原本的纯 Markdown 提示改为 HTML 卡片样式（边框 + 圆角 + 内边距）。
  - 卡片背景改为低透明度黄色：`background: rgba(245, 158, 11, 0.12)`，并保留公式说明内容不变。

### 2026-04-16：板块1 卡片间距调整

- 你的补充需求：
  - 增大黄色提示卡片与下方功能区域的间距（当前贴在一起）。
- 我做的修改（已在 `week7.py` 中实现）：
  - 在板块1黄色提示卡片样式中增加底部外边距：`margin-bottom: 18px`。

### 2026-04-16：顶部标签标题精简

- 你的补充需求：
  - 顶部可点击子页面标题仅保留“板块1/2/3/4”，不带后缀说明。
- 我做的修改（已在 `week7.py` 中实现）：
  - 将 `st.tabs(...)` 文案从“板块 x：说明”统一改为：`板块1`、`板块2`、`板块3`、`板块4`。

### 2026-04-16：板块2 提示卡配色统一

- 你的补充需求：
  - 板块2中三个提示卡（RNN 记忆机制、超参数影响、学习效果判断）在保留格式不变的前提下，改成和板块1一致的低透明度黄色。
- 我做的修改（已在 `week7.py` 中实现）：
  - 将这三张卡片样式统一改为：
    - 边框：`border: 1px solid #F59E0B`
    - 背景：`background: rgba(245, 158, 11, 0.12)`
  - 保留了原有文案、间距和结构（含 `margin-top/margin-bottom`）不变。

### 2026-04-16：板块3 初学者原理提示卡

- 你的补充需求：
  - 给板块3也加入与板块2同格式的文字提示，简洁解释：
    - BERT 如何利用 `[MASK]` 左右上下文（双向注意力机制）
    - GPT-2 如何按从左到右顺序续写（单向自回归约束）
- 我做的修改（已在 `week7.py` 中实现）：
  - 在 `transformers_tab_3()` 顶部新增两张黄色提示卡（样式同板块2）：
    - `BERT 如何利用 [MASK] 两侧上下文（双向注意力）`
    - `GPT-2 如何从左到右续写（单向自回归约束）`
  - 采用简洁表述，面向初学者说明两种预训练/生成机制的核心差异。

### 2026-04-16：网页主标题调整

- 你的补充需求：
  - 将整个网页大标题改为：`语言模型训练与对比分析平台`。
- 我做的修改（已在 `week7.py` 中实现）：
  - 在 `main()` 中将 `st.title(...)` 更新为 `语言模型训练与对比分析平台`。

### 2026-04-16：部署准备（GitHub + Streamlit Cloud）

- 你的新需求：
  - 通过 GitHub 和 Streamlit Cloud 部署整个程序，并补充 `requirements.txt`。
- 我做的修改：
  - 新增 `requirements.txt`，包含当前项目依赖：
    - `streamlit`
    - `nltk`
    - `torch`
    - `transformers`

### 2026-04-16：板块1 Reuters 输入框显示逻辑优化

- 你的补充需求：
  - 选择“使用 NLTK Reuters”后，输入框不再显示默认文本；优先展示 Reuters 实际内容（框大小不变，超长可滚动），若不可行则清空输入框。
- 我做的修改（已在 `week7.py` 中实现）：
  - 新增 `_load_reuters_preview()`（带缓存）加载 Reuters 文本预览。
  - 在 Reuters 选项下，文本框优先显示 Reuters 实际文本（长文本可在 `text_area` 内滚动）。
  - 若 Reuters 预览加载失败，输入框改为置空并给出提示，不再回退显示默认文本。
  - 训练时优先使用输入框当前内容；若为空，再兜底尝试加载 Reuters，仍失败则回退到默认示例文本。

### 2026-04-16：板块1 语料来源交互顺序修正

- 你的补充问题：
  - 之前切换 Reuters 后，文本框不会立即变化，只有点击“构建模型”才显示 Reuters 文本。
  - 希望把“语料来源”位置移动到“输入框”和“构建模型按钮”中间。
- 我做的修改（已在 `week7.py` 中实现）：
  - 移除了板块1的 `st.form` 提交流程，改为普通控件 + 普通按钮，避免表单导致的“切换不立即刷新”。
  - 按你要求调整顺序为：输入框 -> 语料来源 -> 构建模型按钮。
  - 使用 `st.session_state["ngram_source"]` 管理来源选择，保证切换后界面能按来源重渲染对应文本。

