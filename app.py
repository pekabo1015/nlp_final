from __future__ import annotations

from pathlib import Path

import streamlit as st

from shared.legacy_runner import run_legacy_streamlit_script


BASE_DIR = Path(__file__).resolve().parent

MODULE_EXAMPLES = {
    "板块 1：中文分词与词性分析": {
        "输入示例": "今天的天气很好，南京长江大桥上有很多行人，有结婚的和尚未结婚的，我们会在里面找人回答自然语言处理问题。",
    },
    "板块 3：语义分析综合平台": {
        "输入示例": (
            'The Boat Race is a side-by-side rowing competition between the University of Oxford '
            '(sometimes referred to as the "Dark Blues") and the University of Cambridge '
            '(sometimes referred to as the "Light Blues"). First held in 1829, the race takes place '
            "on the 4.2-mile (6.8 km) Championship Course, between Putney and Mortlake on the River Thames "
            "in south-west London. The rivalry is a major point of honour between the two universities; "
            "it is followed throughout the United Kingdom and broadcast worldwide. Oxford went into the 2016 "
            "race as champions, having won the 2015 race by a margin of six lengths, but Cambridge led overall "
            "with 81 victories to Oxford's 79 (excluding the 1877 race, officially a dead heat though claimed "
            "as a victory by the Oxford crew). It was the first time in the history of The Boat Race that all "
            "four senior races – the men's, women's, men's reserves' and women's reserves' – were held on the "
            "same day and on the same course along the Tideway. Prior to 2015, the women's race, which first "
            "took place in 1927, was usually held at the Henley Boat Races along the 2,000-metre (2,200 yd) "
            "course; on at least two occasions in the interwar period, the women competed on the Thames between "
            "Chiswick and Kew. Oxford went into the race as reigning champions, having won the 2015 race by six "
            "and a half lengths, with Cambridge leading 41–29 overall. For the fourth year, the men's race was "
            "sponsored by BNY Mellon while the women's race was sponsored by BNY Mellon's subsidiary, Newton "
            "Investment Management. In January 2016, it was announced that the sponsors would donate the title "
            "sponsorship to Cancer Research UK and that the event was to be retitled \"The Cancer Research UK Boat Races\". "
            "There is no monetary award for winning the race, as the journalist Roger Alton notes: \"It's the last great "
            "amateur event: seven months of pain for no prize money\". On Sunday 27 March, the women's race started at "
            "3:10 p.m. British Summer Time, the women's reserve race (between Oxford's Osiris and Cambridge's Blondie) "
            "at 3:25 p.m., the men's reserves' race (between Oxford's Isis and Cambridge's Goldie) fifteen minutes later "
            "and the men's race a further half-hour after that at 4:10 pm. The men's race was umpired for the fifth time "
            "by Simon Harris, who had overseen the inaugural Tideway running of the Women's Boat Race in 2015. He rowed "
            "for Cambridge in the 1982 and 1983 races and was most recently umpire for the men's race in 2010 Rob Clegg, "
            "umpire for the 2011 race and three-time Oxford Blue, took charge of the women's race. The men's and women's "
            "reserves' races were umpired by Sarah Winckless and Judith Packer respectively, Winckless becoming the first-ever "
            "female official of a men's race. Although around 250,000 spectators were expected to line the banks of the river, "
            "engineering works and poor weather reduced the attendance. The event was broadcast live in the United Kingdom on "
            "the BBC. Numerous broadcasters worldwide also showed the main races, including SuperSport across Africa, the EBU "
            "across Europe, SKY México across Central America, TSN in Canada and Fox Sports in Australia. It was also streamed "
            "live on BBC Online."
        ),
    },
    "板块 8：机器翻译与 BLEU 评测": {
        "示例 1（英文）": "It rains cats and dogs.",
        "示例 2（英文）": "He likes the book that I see today.",
        "示例 3（中文）": "音乐喜欢她非常",
        "示例 4（中文）": "她十分喜爱音乐",
    },
    "板块 9：舆情情感分析系统": {
        "好评": "挺好的，方便携带，不易撕破，物有所值，收到货感觉还是挺好的，家里有大包，就是想着出去的时候包包里面装包小的方便。纸巾很厚实没有什么异味，很好",
        "差评": "太难用了，纸张很薄，性价比低",
        "整体满意夹带抱怨": "挺好的，方便携带，不易撕破，收到货感觉还是挺好的，纸巾很厚实没有什么异味，就是有点贵，性价比不高",
        "显式": "这屏幕画质太垃圾了",
        "隐式": "在太阳底下根本看不清屏幕上的字",
    },
}

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

    examples = MODULE_EXAMPLES.get(module["title"])
    if examples:
        with st.expander("示例文本（可复制）", expanded=False):
            for label, text in examples.items():
                st.text_area(label, value=text, height=120, key=f"example_{module['title']}_{label}")

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
