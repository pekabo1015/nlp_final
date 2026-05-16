import math
from collections import Counter

import nltk
import streamlit as st
import torch
import torch.nn as nn
try:
    from transformers import pipeline
except Exception:  # transformers 可能未安装/环境不完整
    pipeline = None
from nltk.tokenize import wordpunct_tokenize


def _tokenize_en(text: str) -> list[str]:
    """NLTK 分词：避免依赖 punkt；返回小写 token 列表。"""
    text = (text or "").strip()
    if not text:
        return []
    tokens = wordpunct_tokenize(text)
    return [t.lower() for t in tokens if t.strip()]


@st.cache_data(show_spinner=False)
def _load_reuters_preview(max_files: int = 50, max_chars: int = 12000) -> tuple[str, str | None]:
    """
    加载 Reuters 预览文本（用于展示在输入框中）：
    - 成功：返回 (preview_text, None)
    - 失败：返回 ("", error_message)
    """
    try:
        from nltk.corpus import reuters

        try:
            file_ids = reuters.fileids()[:max_files]
        except LookupError:
            nltk.download("reuters", quiet=True)
            file_ids = reuters.fileids()[:max_files]

        raw = " ".join(reuters.raw(fid) for fid in file_ids)
        raw = raw.strip()
        if max_chars > 0 and len(raw) > max_chars:
            raw = raw[:max_chars]
        return raw, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def build_ngram_trigram(tokens: list[str]) -> dict:
    """
    基于“整段语料作为一条序列”的方式构建 trigram 模型：
    P(w_i | w_{i-2}, w_{i-1}) = C(w_{i-2}, w_{i-1}, w_i) / C(w_{i-2}, w_{i-1})
    """
    # 用固定 start/end token 让 input 计算方式保持一致
    seq = ["<s>", "<s>"] + tokens + ["</s>"]

    trigram_counts = Counter()
    bigram_prefix_counts = Counter()

    for i in range(len(seq) - 2):
        w1, w2, w3 = seq[i], seq[i + 1], seq[i + 2]
        trigram_counts[(w1, w2, w3)] += 1
        bigram_prefix_counts[(w1, w2)] += 1

    # 加一平滑的词表大小：统计语料中出现过的 token，再加 start/end
    vocab = set(tokens)
    vocab_size = len(vocab) + 2  # "<s>" 和 "</s>"

    word_freq = Counter(tokens)  # 只统计真实词频，不含 <s>/<\s>

    return {
        "n": 3,
        "tokens": tokens,
        "trigram_counts": trigram_counts,
        "bigram_prefix_counts": bigram_prefix_counts,
        "vocab_size": vocab_size,
        "word_freq": word_freq,
    }


def sentence_joint_probability(
    tokens: list[str],
    model: dict,
    laplace_smoothing: bool,
) -> dict:
    """
    计算句子联合概率：
    对 input=<s>,<s>,w1,...,wm,</s>：
      P = Π_{t=2..T-1} P(seq[t] | seq[t-2], seq[t-1])
    以 log-prob 展示，避免严重下溢。
    """
    seq = ["<s>", "<s>"] + tokens + ["</s>"]

    trigram_counts: Counter = model["trigram_counts"]
    bigram_prefix_counts: Counter = model["bigram_prefix_counts"]
    vocab_size: int = model["vocab_size"]

    log_prob = 0.0
    zero_events = []
    factors = []

    # t 指向“要预测的 token”
    for t in range(2, len(seq)):
        w_prev2, w_prev1 = seq[t - 2], seq[t - 1]
        w = seq[t]
        prefix = (w_prev2, w_prev1)
        trigram = (w_prev2, w_prev1, w)

        c_prefix = bigram_prefix_counts.get(prefix, 0)
        c_trigram = trigram_counts.get(trigram, 0)

        if laplace_smoothing:
            denom = c_prefix + vocab_size
            num = c_trigram + 1
            p = num / denom if denom > 0 else 0.0
            # laplace 下不应为 0，但仍做保护
            if p <= 0:
                zero_events.append({"type": "non_positive_prob", "prefix": prefix, "trigram": trigram})
                p = 1e-300
            log_p = math.log(p)
            factors.append(
                {
                    "prefix": " ".join(prefix),
                    "trigram": " ".join(trigram),
                    "c_prefix": c_prefix,
                    "c_trigram": c_trigram,
                    "p": p,
                    "mode": "laplace",
                }
            )
            log_prob += log_p
        else:
            # 未平滑：若 trigram 未出现或 prefix 未出现，则条件概率为 0
            if c_prefix == 0 or c_trigram == 0:
                zero_events.append({"type": "zero_event", "prefix": prefix, "trigram": trigram})
                p = 0.0
                factors.append(
                    {
                        "prefix": " ".join(prefix),
                        "trigram": " ".join(trigram),
                        "c_prefix": c_prefix,
                        "c_trigram": c_trigram,
                        "p": 0.0,
                        "mode": "unsmoothed",
                    }
                )
                # 联合概率直接为 0
                return {
                    "log_prob": float("-inf"),
                    "prob": 0.0,
                    "zero_events": zero_events,
                    "factors": factors,
                }
            p = c_trigram / c_prefix
            log_prob += math.log(p)
            factors.append(
                {
                    "prefix": " ".join(prefix),
                    "trigram": " ".join(trigram),
                    "c_prefix": c_prefix,
                    "c_trigram": c_trigram,
                    "p": p,
                    "mode": "unsmoothed",
                }
            )

    # exp(log_prob) 会严重下溢，因此也展示 log_prob
    prob = math.exp(log_prob) if log_prob > -745 else 0.0
    return {
        "log_prob": log_prob,
        "prob": prob,
        "zero_events": zero_events,
        "factors": factors,
    }


def ngram_tab_1() -> None:
    st.subheader("n 元语言模型与数据平滑（第 1 标签页：Trigram + Add-one）")
    st.write("使用 NLTK 从语料构建 Trigram 统计模型，并计算输入句子的联合生成概率。")
    st.markdown(
        "<div style='border:1px solid #F59E0B; padding:14px 16px; border-radius:10px; background:rgba(245, 158, 11, 0.12); margin-bottom:18px'>"
        "<b>关键提示：加一平滑（Add-one / Laplace）如何给“未见事件”分配概率</b><br/>"
        "在我们的条件概率定义中：<br/>"
        "<code>P(w_t | w_{t-2}, w_{t-1}) = C(w_{t-2}, w_{t-1}, w_t) / C(w_{t-2}, w_{t-1})</code>（不平滑，MLE）。<br/><br/>"
        "当某个三元组（或其前缀 <code>w_{t-2}, w_{t-1}</code> 下的某个后续词）在语料中从未出现时，MLE 会给出概率 0。为避免这种情况，开启加一平滑后：<br/>"
        "<code>P(w_t | w_{t-2}, w_{t-1}) = (C(w_{t-2}, w_{t-1}, w_t)+1) / (C(w_{t-2}, w_{t-1}) + V)</code>，其中 <code>V</code> 是词表大小。<br/><br/>"
        "你可以把分母中新增的 <code>+V</code> 理解为“概率余量” δ 的来源：总新增了 <code>V</code> 份（每个候选后续词都 +1）。因此每个未见候选词（对应 <code>C(...)=0</code>）都会均匀获得：<br/>"
        "<code>δ_未见 = 1 / (C(w_{t-2}, w_{t-1}) + V)</code>。<br/>"
        "而所有出现过的三元组也会相应地从 MLE 概率做一个折扣（因为分母变大）。"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 1) 加载英文语料并构建 Trigram 模型")
    default_text = (
        "Reuters is a news agency that provides financial information. "
        "Traders use Reuters data to make decisions."
    )

    source = st.session_state.get("ngram_source", "手动输入")

    if source == "手动输入":
        corpus_text = st.text_area("输入语料（英文，建议一段短文本）", value=default_text, height=160)
    else:
        # 优先展示 Reuters 实际文本；文本框高度不变，超长内容可在框内滚动查看
        reuters_text, reuters_err = _load_reuters_preview(max_files=50, max_chars=12000)
        if reuters_err:
            st.warning(f"Reuters 预览加载失败，输入框将置空：{reuters_err}")
        reuters_value = reuters_text if reuters_text else ""
        corpus_text = st.text_area(
            "Reuters 内容（已自动加载，可在框内滚动查看）",
            value=reuters_value,
            height=160,
        )

    st.radio(
        "语料来源",
        ["手动输入", "使用 NLTK Reuters（如已下载）"],
        horizontal=True,
        key="ngram_source",
    )

    train_clicked = st.button("构建 Trigram 模型（统计词频）", key="build_trigram_btn")

    if train_clicked:
        with st.spinner("正在分词与统计 n-gram 计数..."):
            tokens = []

            if source == "使用 NLTK Reuters（如已下载）":
                # 优先使用输入框里展示的 Reuters 文本，与界面保持一致
                tokens = _tokenize_en(corpus_text)
                if not tokens:
                    # 若输入框为空（例如预览加载失败），再兜底尝试一次 Reuters 读取
                    reuters_text, reuters_err = _load_reuters_preview(max_files=50, max_chars=0)
                    if reuters_text:
                        tokens = _tokenize_en(reuters_text)
                    else:
                        e = reuters_err or "未知错误"
                        st.warning(
                            "Reuters 加载失败（可能是网络受限或本地缺少 NLTK 语料），"
                            f"已回退为默认文本。错误：{e}"
                        )
                        tokens = _tokenize_en(default_text)
                if not tokens:
                    st.warning(
                        "Reuters 文本分词结果为空，已回退为默认文本。"
                    )
                    tokens = _tokenize_en(default_text)
            else:
                tokens = _tokenize_en(corpus_text)

            if not tokens:
                st.error("语料为空或分词结果为空，请检查输入文本。")
                return

            st.session_state["tri_model"] = build_ngram_trigram(tokens)
            st.session_state["tri_model_built"] = True

    if not st.session_state.get("tri_model_built"):
        st.info("请先在上方点击“构建 Trigram 模型”。")
        return

    model = st.session_state["tri_model"]

    st.markdown("### 2) 模型统计信息（词频）")
    st.caption("词频统计基于语料分词结果（不包含模型的 `<s>` 和 `</s>`）。")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("词表大小（用于加一平滑）", model["vocab_size"])
    with col2:
        st.metric("语料 token 数", len(model["tokens"]))
    with col3:
        st.metric("不同 trigram 数", len(model["trigram_counts"]))

    top_k = st.slider("展示 Top-K 词频", min_value=10, max_value=100, value=30, step=5)
    wf = model["word_freq"]
    most = wf.most_common(top_k)
    st.dataframe(
        [{"word": w, "count": c} for w, c in most],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### 3) 输入句子并计算联合生成概率")

    input_sentence = st.text_area(
        "输入英文句子（会在句首加入 `<s> <s>`，并在句尾加入 `</s>`）",
        value="Reuters provides financial information",
        height=90,
    )

    use_laplace = st.checkbox("开启加一平滑（Laplace / Add-one）用于当前概率显示", value=False)
    calc_clicked = st.button("计算生成概率（联合概率）", type="primary")

    if calc_clicked:
        tokens_in = _tokenize_en(input_sentence)
        if not tokens_in:
            st.error("输入句子分词为空，请检查输入。")
            return

        # 未平滑：用于零概率事件检测与“对比显示”
        unsmoothed = sentence_joint_probability(tokens_in, model, laplace_smoothing=False)
        # 加一平滑：用于对比
        laplace = sentence_joint_probability(tokens_in, model, laplace_smoothing=True)

        has_zero_event = len(unsmoothed["zero_events"]) > 0

        # 顶部结论展示（严格跟随复选框选择；若检测到零概率事件则额外展示对比）
        primary = laplace if use_laplace else unsmoothed
        secondary = unsmoothed if use_laplace else laplace
        primary_label = "加一平滑联合概率" if use_laplace else "未平滑联合概率"
        secondary_label = "未平滑联合概率" if use_laplace else "加一平滑联合概率"

        if has_zero_event:
            cA, cB = st.columns(2)
            with cA:
                st.metric(
                    primary_label,
                    f"{primary['prob']:.6g}" if primary["prob"] > 0 else "0",
                    delta=None,
                )
                st.caption(f"logP = {primary['log_prob']}")
            with cB:
                st.metric(
                    secondary_label,
                    f"{secondary['prob']:.6g}" if secondary["prob"] > 0 else "0",
                    delta=None,
                )
                st.caption(f"logP = {secondary['log_prob']}")
        else:
            st.metric(
                primary_label,
                f"{primary['prob']:.6g}" if primary["prob"] > 0 else "0",
                delta=None,
            )
            st.caption(f"logP = {primary['log_prob']}")

        if has_zero_event:
            st.warning("检测到输入句子中存在语料未出现的 trigram（未平滑为零概率事件）。下面展示对比信息。")
            # 展示前几个零事件细节
            preview_n = min(10, len(unsmoothed["zero_events"]))
            st.write("零概率事件预览（prefix -> trigram）：")
            for e in unsmoothed["zero_events"][:preview_n]:
                prefix = e["prefix"]
                trigram = e["trigram"]
                st.write(f"- `{prefix[0]} {prefix[1]}` -> `{trigram[0]} {trigram[1]} {trigram[2]}`")
        else:
            st.success("未检测到零概率事件：未平滑与模型词表覆盖一致。")

        # 当前显示结果说明（只在检测到零概率事件时才强制对比）
        st.markdown("### 4) 当前显示结果（按复选框）")
        if use_laplace and has_zero_event:
            st.info("已开启加一平滑：顶部显示加一平滑联合概率，并展示未平滑对比。")
        elif use_laplace and not has_zero_event:
            st.info("已开启加一平滑：顶部显示加一平滑联合概率。")
        elif (not use_laplace) and has_zero_event:
            st.info("未开启加一平滑：顶部显示未平滑联合概率，并展示加一平滑对比。")
        else:
            st.info("未开启加一平滑：顶部显示未平滑联合概率。")

        # 展示因子分解表（通常较长，放到展开里）
        with st.expander("查看联合概率因子分解（每个位置的条件概率）", expanded=False):
            st.caption("下表分别展示未平滑与加一平滑每个条件概率 P(w_t | w_{t-2}, w_{t-1})。")
            # 未平滑可能在第一处零事件时直接返回，因此 factors 可能较短
            uns_factors = unsmoothed["factors"]
            lap_factors = laplace["factors"]
            st.write("未平滑 factors（部分）：")
            st.dataframe(uns_factors[: min(50, len(uns_factors))], use_container_width=True)
            st.write("加一平滑 factors（部分）：")
            st.dataframe(lap_factors[: min(50, len(lap_factors))], use_container_width=True)


class CharRNNLM(nn.Module):
    """简单字符级 RNN 语言模型：Embedding + RNN + Linear。"""

    def __init__(self, vocab_size: int, hidden_size: int, emb_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        emb = self.embedding(x)  # [B, T, E]
        out, h_n = self.rnn(emb, h0)  # out: [B, T, H]
        logits = self.fc(out)  # [B, T, V]
        return logits, h_n


def _build_char_dataset(text: str):
    text = text or ""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    idx = [stoi[ch] for ch in text]
    return stoi, itos, idx


def _generate_text(
    model: CharRNNLM,
    stoi: dict[str, int],
    itos: dict[int, str],
    seed: str,
    gen_len: int = 50,
    temperature: float = 1.0,
) -> str:
    if not seed:
        seed = " "
    device = next(model.parameters()).device
    model.eval()

    # 若 seed 含未见字符，跳过未见字符；全部未见则回退到词表第一个字符
    seed_ids = [stoi[ch] for ch in seed if ch in stoi]
    if not seed_ids:
        seed_ids = [0]
        seed = itos[0]

    generated = list(seed)
    h = None

    with torch.no_grad():
        # 先“喂入” seed，让隐藏状态对齐上下文
        x = torch.tensor(seed_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, h = model(x, h)
        last_token = x[:, -1:]

        for _ in range(gen_len):
            logits, h = model(last_token, h)  # [1,1,V]
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(step_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
            nid = int(next_id.item())
            generated.append(itos[nid])
            last_token = next_id

    return "".join(generated)


def rnn_tab_2() -> None:
    st.subheader("从零训练 RNN 语言模型（第 2 标签页：Character-level RNN-LM）")
    st.write("使用 PyTorch 在小语料上进行字符级自回归训练：用前 t-1 个字符预测第 t 个字符。")

    st.markdown(
        "<div style='border:1px solid #F59E0B; padding:14px 16px; border-radius:10px; background:rgba(245, 158, 11, 0.12)'>"
        "<b>RNN 在“记住过去信息”</b><br/>"
        "这个模型的核心在于隐藏状态 <i>h</i>：每一步处理到新字符时，都会把“历史上下文”压缩到 <i>h</i> 里，"
        "从而让预测不仅依赖当前字符，也依赖之前的字符序列。<br/>"
        "因此，后面的超参数（尤其是 Hidden Size）会直接影响模型“记忆容量”。"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='border:1px solid #F59E0B; padding:14px 16px; border-radius:10px; background:rgba(245, 158, 11, 0.12); margin-top:10px'>"
        "<b>超参数如何影响训练效果</b><br/>"
        "• <b>Hidden Size</b>：隐藏维度越大，模型能表达的模式越复杂（记忆容量更强）；但也可能更慢、在小语料上更容易过拟合。"
        "<br/>"
        "• <b>Epochs</b>：训练轮数越多，模型看到同样数据的次数越多；Loss 通常会下降，但太多轮次可能开始“死记硬背”。"
        "<br/>"
        "• <b>Learning Rate</b>：学习率决定每次参数更新的步子大小；太大可能导致训练不稳定，太小则收敛太慢。"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='border:1px solid #F59E0B; padding:14px 16px; border-radius:10px; background:rgba(245, 158, 11, 0.12); margin-top:10px; margin-bottom:18px'>"
        "<b>怎么判断模型是否“学到了输入序列的模式”</b><br/>"
        "• 看 Loss 曲线：训练时 Loss 若明显下降，通常说明模型在学习语料中更常见的字符转移规律。"
        "<br/>"
        "• 看生成文本：生成出来的字符序列往往会呈现语料的风格（例如标点频率、常见片段复现、重复韵脚/结构等）。"
        "<br/>"
        "如果生成文本仍然“随机噪声”，可以尝试增加 Hidden Size 或 Epochs，或把 Learning Rate 调得更合适。"
        "</div>",
        unsafe_allow_html=True,
    )

    default_corpus = (
        "Two roads diverged in a yellow wood,\n"
        "And sorry I could not travel both,\n"
        "And be one traveler, long I stood.\n"
    )
    st.markdown("<div style='font-size:18px; font-weight:600; margin-bottom:2px'>输入自定义短语料</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; color:#6B7280; margin-bottom:8px'>(英文，建议几十到几百字符)</div>", unsafe_allow_html=True)
    corpus_text = st.text_area("", value=default_corpus, height=180)

    c1, c2, c3 = st.columns(3)
    with c1:
        hidden_size = st.slider("Hidden Size", min_value=16, max_value=128, value=64, step=16)
    with c2:
        epochs = st.slider("Epochs", min_value=10, max_value=200, value=80, step=10)
    with c3:
        lr = st.slider("Learning Rate", min_value=0.001, max_value=0.2, value=0.02, step=0.001)

    train_clicked = st.button("开始训练", type="primary", key="rnn_train_btn")

    if train_clicked:
        if len(corpus_text) < 20:
            st.error("语料过短，请至少输入 20 个字符。")
            return

        stoi, itos, idx = _build_char_dataset(corpus_text)
        vocab_size = len(stoi)
        if vocab_size < 2:
            st.error("语料字符种类不足，至少需要 2 个不同字符。")
            return

        # 自回归训练样本：x = text[:-1], y = text[1:]
        x_ids = torch.tensor(idx[:-1], dtype=torch.long).unsqueeze(0)  # [1, T]
        y_ids = torch.tensor(idx[1:], dtype=torch.long).unsqueeze(0)  # [1, T]

        model = CharRNNLM(vocab_size=vocab_size, hidden_size=hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        chart_placeholder = st.empty()
        status_placeholder = st.empty()

        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x_ids)  # [1, T, V]
            loss = criterion(logits.reshape(-1, vocab_size), y_ids.reshape(-1))
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            chart_placeholder.line_chart({"loss": losses})

            if ep == 1 or ep % 10 == 0 or ep == epochs:
                status_placeholder.info(f"训练中：Epoch {ep}/{epochs}，Loss = {losses[-1]:.6f}")

        st.success("训练完成。")

        # 保存模型与词表到 session_state，供后续生成使用
        st.session_state["rnn_model"] = model
        st.session_state["rnn_stoi"] = stoi
        st.session_state["rnn_itos"] = itos
        st.session_state["rnn_trained"] = True
        st.session_state["rnn_vocab_size"] = vocab_size
        st.session_state["rnn_last_loss"] = losses[-1]

    st.markdown("---")
    st.markdown("### 生成文本")
    if not st.session_state.get("rnn_trained"):
        st.info("请先点击“开始训练”，训练完成后可在此输入 Seed 生成文本。")
        return

    st.caption(
        f"最近一次训练完成：vocab={st.session_state.get('rnn_vocab_size')}，"
        f"last_loss={st.session_state.get('rnn_last_loss'):.6f}"
    )

    seed = st.text_input("起始字符（Seed）", value="And ")
    gen_len = st.slider("生成长度（字符数）", min_value=20, max_value=200, value=50, step=10)
    temperature = st.slider("采样温度 Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

    if st.button("生成文本", key="rnn_generate_btn"):
        model = st.session_state["rnn_model"]
        stoi = st.session_state["rnn_stoi"]
        itos = st.session_state["rnn_itos"]
        output = _generate_text(
            model=model,
            stoi=stoi,
            itos=itos,
            seed=seed,
            gen_len=gen_len,
            temperature=temperature,
        )
        st.text_area("生成结果", value=output, height=180)


def transformers_tab_3() -> None:
    st.subheader("预训练架构对比（第 3 标签页：Masked LM vs. Causal LM）")
    st.write("使用 Hugging Face `transformers` 的 `pipeline` 分别演示 BERT（Masked LM）与 GPT-2（Causal LM）的生成机制差异。")
    st.markdown(
        "<div style='border:1px solid #F59E0B; padding:14px 16px; border-radius:10px; background:rgba(245, 158, 11, 0.12)'>"
        "<b>BERT 如何利用 [MASK] 两侧上下文（双向注意力）</b><br/>"
        "在 Masked LM 任务中，BERT 会同时参考 <code>[MASK]</code> 左右两侧的词来预测缺失词，"
        "因此它擅长做“补空”这类需要双向上下文的信息整合。"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='border:1px solid #F59E0B; padding:14px 16px; border-radius:10px; background:rgba(245, 158, 11, 0.12); margin-top:10px; margin-bottom:18px'>"
        "<b>GPT-2 如何从左到右续写（单向自回归约束）</b><br/>"
        "GPT-2 在生成第 <i>t</i> 个词时，只能看见前面已经给定或已生成的词（不能看右侧未来词），"
        "这就是单向自回归约束，所以它天然适合“接着往下写”。"
        "</div>",
        unsafe_allow_html=True,
    )

    if pipeline is None:
        st.error("未检测到可用的 `transformers` 包或初始化失败，请先安装 `transformers` 再使用板块 3。")
        return

    # ---- pipeline 缓存，避免每次刷新都重复下载/初始化 ----
    if "bert_fillmask_pipe" not in st.session_state:
        try:
            st.session_state["bert_fillmask_pipe"] = pipeline(
                "fill-mask",
                model="bert-base-uncased",
                tokenizer="bert-base-uncased",
            )
        except Exception as e:
            st.error(f"BERT pipeline 初始化失败：{type(e).__name__}: {e}")
            return
    if "gpt2_textgen_pipe" not in st.session_state:
        try:
            st.session_state["gpt2_textgen_pipe"] = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
            )
        except Exception as e:
            st.error(f"GPT-2 pipeline 初始化失败：{type(e).__name__}: {e}")
            return

    bert_col, gpt_col = st.columns(2)

    # ---------------- BERT ----------------
    with bert_col:
        st.markdown("### BERT（Masked Language Modeling）")
        st.caption("输入需要包含 `[MASK]`。模型会对 `[MASK]` 位置进行 Top-5 预测。")

        if "bert_table" not in st.session_state:
            st.session_state["bert_table"] = None
        if "bert_top1_sequence" not in st.session_state:
            st.session_state["bert_top1_sequence"] = None

        bert_default = "The man went to the [MASK] to buy some milk."
        bert_input = st.text_area("输入（示例：`[MASK]` 标记缺失词）", value=bert_default, height=110)
        bert_clicked = st.button("BERT 预测 Top-5", key="bert_top5_btn", type="primary")

        if bert_clicked:
            if "[MASK]" not in bert_input:
                st.error("请在句子中包含 `[MASK]` 标记，例如：`... to the [MASK] ...`")
            else:
                try:
                    preds = st.session_state["bert_fillmask_pipe"](bert_input, top_k=5)
                    # preds 通常是 List[dict]
                    if isinstance(preds, dict):
                        preds = [preds]

                    table = []
                    for p in preds[:5]:
                        token = p.get("token_str", "")
                        score = p.get("score", 0.0)
                        table.append({"候选词": token, "概率": float(score)})

                    st.session_state["bert_table"] = table
                    if preds:
                        st.session_state["bert_top1_sequence"] = preds[0].get("sequence", "")
                except Exception as e:
                    st.error(f"BERT 推理失败：{type(e).__name__}: {e}")

        # 冻结展示：无论你点 BERT 还是 GPT-2，这里都会显示最近一次计算结果
        if st.session_state.get("bert_table"):
            st.dataframe(st.session_state["bert_table"], use_container_width=True, hide_index=True)
        if st.session_state.get("bert_top1_sequence"):
            st.caption("Top-1 填充后的序列：")
            st.code(st.session_state["bert_top1_sequence"], language="text")

    # ---------------- GPT-2 ----------------
    with gpt_col:
        st.markdown("### GPT-2（Causal Language Modeling）")
        st.caption("输入一个前缀 Prompt 后，从左到右自回归生成。这里截取后续的 20 个单词。")

        if "gpt2_cont20" not in st.session_state:
            st.session_state["gpt2_cont20"] = None

        gpt_prompt_default = "Once upon a time"
        gpt_prompt = st.text_area("Prompt（前缀提示词）", value=gpt_prompt_default, height=110)
        gpt_clicked = st.button("GPT-2 生成 20 个单词", key="gpt2_gen_btn", type="primary")

        if gpt_clicked:
            if not gpt_prompt.strip():
                st.error("Prompt 不能为空。")
            else:
                try:
                    # 生成足够多的 token，后处理为“后续 20 个单词”
                    gen_kwargs = {
                        "max_new_tokens": 120,
                        "do_sample": True,
                        "temperature": 0.9,
                        "top_p": 0.95,
                    }
                    out = st.session_state["gpt2_textgen_pipe"](gpt_prompt, **gen_kwargs)
                    if not out:
                        st.error("GPT-2 没有返回生成结果。")
                    else:
                        gen_text = out[0].get("generated_text", "")

                        # generated_text 通常包含 prompt；稳妥去掉 prompt 前缀
                        continuation = gen_text
                        if gen_text.startswith(gpt_prompt):
                            continuation = gen_text[len(gpt_prompt) :]

                        words = continuation.strip().split()
                        cont20 = " ".join(words[:20])

                        st.session_state["gpt2_cont20"] = cont20
                        if len(words) < 20:
                            st.warning(f"生成到的后续单词不足 20 个（实际 {len(words)} 个）。")
                except Exception as e:
                    st.error(f"GPT-2 推理失败：{type(e).__name__}: {e}")

        # 冻结展示：无论你点 BERT 还是 GPT-2，这里都会显示最近一次计算结果
        if st.session_state.get("gpt2_cont20") is not None:
            st.text_area("生成结果（后续 20 个单词）", value=st.session_state["gpt2_cont20"], height=160)
        else:
            st.info("点击右侧按钮生成后，这里会显示最近一次结果。")

    st.markdown(
        "#### 初学者小结\n"
        "- BERT（Masked LM）= 在给定上下文下“补全缺失词”。你只改 `[MASK]` 位置，其余词不必是自回归顺序生成。\n"
        "- GPT-2（Causal LM）= 从左到右生成，预测下一词依赖已经生成的历史。"
    )


def ppl_tab_4() -> None:
    st.subheader("语言模型评价（第 4 标签页：GPT-2 困惑度 PPL 计算）")
    st.write("输入多段测试句子，基于 GPT-2 计算每句交叉熵损失（Cross-Entropy Loss）与困惑度（PPL = exp(Loss)）。")

    if pipeline is None:
        st.error("未检测到可用的 `transformers` 包或初始化失败，请先安装 `transformers` 再使用板块 4。")
        return

    default_eval_text = (
        "The weather is nice today.\n"
        "I enjoy reading books about natural language processing.\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "Milk quantum banana swiftly keyboard under the triangle moon.\n"
        "Florp snizzle wug blanter zorp into crinkly noodle vapor."
    )
    eval_text = st.text_area(
        "输入测试句子（每行一条；支持多行）",
        value=default_eval_text,
        height=180,
    )
    st.caption("默认示例包含 3 条语法和语义都较自然的英文句子，以及 2 条随机拼词的非自然句，便于直观看到 PPL 差异。")

    calc_clicked = st.button("计算每句 PPL", key="ppl_calc_btn", type="primary")

    if not calc_clicked:
        return

    # 复用板块3已缓存的 gpt2 pipeline；若不存在则初始化
    if "gpt2_textgen_pipe" not in st.session_state:
        try:
            st.session_state["gpt2_textgen_pipe"] = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
            )
        except Exception as e:
            st.error(f"GPT-2 pipeline 初始化失败：{type(e).__name__}: {e}")
            return

    gpt2_pipe = st.session_state["gpt2_textgen_pipe"]
    model = gpt2_pipe.model
    tokenizer = gpt2_pipe.tokenizer
    model.eval()

    # 每行一条句子；过滤空行
    sentences = [line.strip() for line in eval_text.splitlines() if line.strip()]
    if not sentences:
        st.error("没有可计算的句子。请至少输入一行非空文本。")
        return

    rows = []
    for sent in sentences:
        try:
            inputs = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = float(outputs.loss.item())
                ppl = float(math.exp(loss))

            rows.append(
                {
                    "句子": sent,
                    "Cross-Entropy Loss": round(loss, 6),
                    "PPL": round(ppl, 6),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "句子": sent,
                    "Cross-Entropy Loss": "ERROR",
                    "PPL": f"{type(e).__name__}: {e}",
                }
            )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("提示：PPL 越低，通常说明该句子在当前语言模型下越“可预测”。")


def main() -> None:
    st.set_page_config(page_title="Week7 语义分析综合测试平台", layout="wide")
    st.title("语言模型训练与对比分析平台")
    st.caption("当前实现：第 1 页 n-gram & Add-one，第 2 页字符级 RNN-LM，第 3 页预训练架构对比，第 4 页 GPT-2 PPL 评价。")

    tabs = st.tabs(
        [
            "板块1",
            "板块2",
            "板块3",
            "板块4",
        ]
    )

    with tabs[0]:
        ngram_tab_1()

    with tabs[1]:
        rnn_tab_2()

    with tabs[2]:
        transformers_tab_3()

    with tabs[3]:
        ppl_tab_4()


if __name__ == "__main__":
    main()

