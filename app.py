from __future__ import annotations

from pathlib import Path

import streamlit as st

from shared.legacy_runner import run_legacy_streamlit_script


BASE_DIR = Path(__file__).resolve().parent

MODULES = [
    {
        "title": "首页",
        "caption": "系统总览、使用说明与云端部署提示。",
        "script": None,
    },
    {
        "title": "板块 1：中文分词与词性分析",
        "caption": "对应 A1，保留原有中文文本处理、多算法分词、词频统计与词性标注。",
        "script": BASE_DIR / "A1" / "streamlit_app.py",
    },
    {
        "title": "板块 2：句法分析系统",
        "caption": "对应 A2，保留依存句法、成分句法、歧义提示与核心论元提取。",
        "script": BASE_DIR / "A2" / "代码2.py",
    },
    {
        "title": "板块 3：语义分析综合平台",
        "caption": "对应 A3，保留 TF-IDF、LSA、Word2Vec、GloVe、FastText 与句子表示实验。",
        "script": BASE_DIR / "A3" / "analysis.py",
    },
    {
        "title": "板块 4：词义消歧与语义角色标注",
        "caption": "对应 A4，保留 Lesk、BERT 上下文词向量、余弦相似度与 SRL 分析。",
        "script": BASE_DIR / "A4" / "week5.py",
    },
    {
        "title": "板块 5：篇章分析与指代消解",
        "caption": "对应 A5，保留 EDU 切分、显式连接词分析与指代消解。",
        "script": BASE_DIR / "A5" / "week6.py",
    },
    {
        "title": "板块 6：语言模型训练与对比分析",
        "caption": "对应 A6，保留 n-gram、RNN、BERT/GPT-2 与 PPL 评价实验。",
        "script": BASE_DIR / "A6" / "week7.py",
    },
    {
        "title": "板块 7：实体关系抽取与知识图谱",
        "caption": "对应 A7，保留实体识别、关系抽取和知识图谱可视化。",
        "script": BASE_DIR / "A7" / "week8.py",
    },
    {
        "title": "板块 8：机器翻译与 BLEU 评测",
        "caption": "对应 A8，保留英中翻译、规则翻译对比与 BLEU 评测。",
        "script": BASE_DIR / "A8" / "week9.py",
    },
    {
        "title": "板块 9：舆情情感分析系统",
        "caption": "对应 A9，保留单文本情感分析、显隐式对比和批量舆情看板。",
        "script": BASE_DIR / "A9" / "week10.py",
    },
]


def render_global_style() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            .app-hero {
                padding: 1.25rem 1.4rem;
                border: 1px solid #dbe5f1;
                border-radius: 16px;
                background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
                margin-bottom: 1rem;
            }
            .app-card {
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 1rem 1.1rem;
                background: #ffffff;
                margin-bottom: 0.8rem;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            }
            .app-muted {
                color: #475569;
                line-height: 1.75;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h2 style="margin:0 0 0.6rem 0;">自然语言处理交互式 Web 分析系统</h2>
            <div class="app-muted">
                本系统基于 Streamlit 构建，将 A1 到 A9 九个自然语言处理实验整合为同一个网页，
                支持在左侧目录中自由切换，并面向 GitHub 与 Streamlit Cloud 在线部署。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("系统说明")
    st.markdown(
        """
        - 左侧边栏用于切换首页和 9 个实验板块。
        - 每个板块都会尽量直接复用你原有的作业代码与交互逻辑。
        - 根目录提供统一入口 `app.py`，便于推送到 GitHub 后直接接入 Streamlit Cloud。
        - 某些模型在云端首次运行时可能需要下载，首次加载时间会比本地更长。
        """
    )

    st.subheader("板块总览")
    for module in MODULES[1:]:
        st.markdown(
            f"""
            <div class="app-card">
                <div style="font-weight:700; margin-bottom:0.35rem;">{module["title"]}</div>
                <div class="app-muted">{module["caption"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("GitHub 与 Streamlit Cloud 部署提示")
    st.markdown(
        """
        - 将当前项目整体推送到同一个 GitHub 仓库。
        - 在 Streamlit Cloud 中选择该仓库，并把主文件设置为 `app.py`。
        - 确保根目录 `requirements.txt` 可完整安装依赖。
        - 如果某些大模型在云端不可稳定下载，系统会在对应板块给出友好提示。
        """
    )


def render_sidebar() -> dict:
    st.sidebar.title("NLP 系统目录")
    selected_title = st.sidebar.radio(
        "选择页面",
        [item["title"] for item in MODULES],
        index=0,
    )

    selected_module = next(item for item in MODULES if item["title"] == selected_title)

    st.sidebar.markdown("---")
    st.sidebar.caption("部署入口")
    st.sidebar.code("streamlit run app.py", language="bash")
    st.sidebar.caption("云端部署目标：GitHub + Streamlit Cloud")

    return selected_module


def render_selected_module(module: dict) -> None:
    st.title(module["title"])
    st.caption(module["caption"])

    if module["script"] is None:
        render_home()
        return

    st.info(f"当前加载脚本：`{module['script'].relative_to(BASE_DIR)}`")
    run_legacy_streamlit_script(module["script"])


def main() -> None:
    st.set_page_config(
        page_title="自然语言处理交互式 Web 分析系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_global_style()
    selected_module = render_sidebar()
    render_selected_module(selected_module)


if __name__ == "__main__":
    main()
