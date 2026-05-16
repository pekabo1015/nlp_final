# Streamlit 入口：在 Cloud 上脚本常在非主线程执行，须先避免第三方库注册 signal 时报错。
import signal

_orig_signal = signal.signal


def _safe_signal(signalnum, handler):
    try:
        return _orig_signal(signalnum, handler)
    except ValueError as exc:
        if "main thread" in str(exc).lower():
            return signal.SIG_DFL
        raise


signal.signal = _safe_signal

import streamlit as st
import pandas as pd

import main2 as core

st.set_page_config(page_title="中文分词 · 多算法对比", layout="wide")

st.title("中文文本处理 · 多算法分词对比")
st.caption("文本规范化、多种分词对比、统计与词性标注（Streamlit 版）")

raw = st.text_area(
    "请输入中文长文本（含歧义句更易看出算法差异）",
    height=160,
    placeholder="例如：结婚的和尚未结婚的、南京市长江大桥、今天天气很好我们正在开发自然语言处理应用。",
)

if st.button("开始处理", type="primary"):
    if not raw.strip():
        st.warning("请先输入文本。")
    else:
        normalized = core.normalize_text(raw)
        st.subheader("文本规范化")
        st.write(normalized or "（空）")
        st.caption(
            "去除特殊符号、全角转半角、繁体转简体"
            + ("（未安装 OpenCC 则跳过繁简）" if not core.OPENCC_AVAILABLE else "")
        )

        if not normalized:
            st.stop()

        if core.JIEBA_AVAILABLE:
            words = core.segment_jieba_default(normalized)
            st.subheader("默认 jieba 分词与词频")
            st.write(" ".join(words) if words else "（无）")
            top_freq = core.get_top_frequencies(words, limit=5)
            if top_freq:
                st.caption("最高频 5 词：" + "，".join(f"{w}({c})" for w, c in top_freq))
                labels = [w for w, _ in top_freq]
                values = [c for _, c in top_freq]
                st.bar_chart(
                    pd.DataFrame({"频次": values}, index=labels),
                    height=220,
                )

            tagged = [
                (x.word, x.flag)
                for x in core.pseg.cut(normalized)
                if x.word.strip()
            ]
            st.subheader("词性标注")
            st.caption("名词红 / 动词蓝 / 形容词绿")
            st.markdown(
                str(core.build_pos_html(tagged)),
                unsafe_allow_html=True,
            )
        else:
            st.error("未安装 jieba：" + (core.JIEBA_IMPORT_ERROR or "请 pip install jieba"))

        st.subheader("多算法分词对比")
        algo_results = core.run_all_segmenters(normalized)
        if not algo_results:
            st.info("当前无可用分词器或全部失败。")
        else:
            cols = st.columns(min(len(algo_results), 2))
            for i, (name, wlist) in enumerate(algo_results):
                with cols[i % len(cols)]:
                    st.markdown(f"**{name}**")
                    st.write(" ".join(wlist) if wlist else "（无结果）")

            st.subheader("统计分析")
            stats_rows = []
            for name, wlist in algo_results:
                s = core.stat_segments(wlist)
                stats_rows.append(
                    {
                        "算法": name,
                        "词数": s["count"],
                        "平均词长": s["avg_len"],
                        "唯一词数": s["unique"],
                    }
                )
            st.dataframe(stats_rows, use_container_width=True, hide_index=True)

            names = [r["算法"] for r in stats_rows]
            counts = [r["词数"] for r in stats_rows]
            if counts:
                st.bar_chart(
                    pd.DataFrame({"词数": counts}, index=names),
                    height=max(200, 40 * len(names)),
                )

            sim_names, sim_matrix = core.build_similarity_matrix(algo_results)
            st.caption("算法间 Jaccard 相似度")
            st.dataframe(
                pd.DataFrame(sim_matrix, index=sim_names, columns=sim_names),
                use_container_width=True,
            )
