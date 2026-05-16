import re
from collections import Counter
from html import escape

from flask import Flask, render_template_string, request
from markupsafe import Markup

try:
    import jieba
    import jieba.posseg as pseg

    JIEBA_AVAILABLE = True
    JIEBA_IMPORT_ERROR = ""
except ImportError as exc:
    JIEBA_AVAILABLE = False
    JIEBA_IMPORT_ERROR = str(exc)

try:
    from opencc import OpenCC

    OPENCC = OpenCC("t2s")
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC = None
    OPENCC_AVAILABLE = False

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>中文分词与词性标注演示</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            margin: 0;
            font-family: "Microsoft YaHei", Arial, sans-serif;
            background: #f5f7fb;
            color: #1f2937;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }
        h1 {
            margin-bottom: 8px;
        }
        .desc {
            margin-bottom: 20px;
            color: #4b5563;
        }
        .input-card, .panel {
            background: #ffffff;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            padding: 18px;
        }
        textarea {
            width: 100%;
            min-height: 180px;
            resize: vertical;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 14px;
            box-sizing: border-box;
            font-size: 15px;
            line-height: 1.7;
        }
        button {
            margin-top: 14px;
            background: #2563eb;
            color: #ffffff;
            border: none;
            padding: 10px 18px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 15px;
        }
        button:hover {
            background: #1d4ed8;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
            margin-top: 22px;
            align-items: start;
        }
        .panel h2 {
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 20px;
        }
        .box {
            min-height: 150px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 14px;
            line-height: 1.9;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .meta {
            margin-top: 12px;
            color: #4b5563;
            font-size: 14px;
        }
        .tags {
            margin-top: 10px;
            line-height: 2.1;
        }
        .tag {
            display: inline-block;
            margin: 0 8px 8px 0;
            padding: 2px 8px;
            border-radius: 999px;
            background: #eef2ff;
            font-size: 14px;
        }
        .noun {
            color: #dc2626;
            font-weight: 700;
        }
        .verb {
            color: #2563eb;
            font-weight: 700;
        }
        .adj {
            color: #16a34a;
            font-weight: 700;
        }
        .other {
            color: #4b5563;
        }
        .legend {
            margin-bottom: 10px;
            font-size: 14px;
            color: #4b5563;
        }
        .warning {
            margin-top: 12px;
            padding: 10px 12px;
            background: #fff7ed;
            border: 1px solid #fdba74;
            border-radius: 10px;
            color: #9a3412;
            font-size: 14px;
        }
        .empty {
            color: #6b7280;
        }
        canvas {
            margin-top: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>中文文本处理 Web 应用</h1>
        <div class="desc">支持文本规范化、中文分词、词频统计可视化与词性标注展示。</div>

        <form method="post" class="input-card">
            <label for="text"><strong>请输入中文长文本</strong></label>
            <textarea id="text" name="text" placeholder="例如：今天天气很好，我们正在开发一个中文自然语言处理 Web 应用。">{{ raw_text }}</textarea>
            <button type="submit">开始处理</button>
        </form>

        <div class="grid">
            <section class="panel">
                <h2>区块1：文本规范化结果</h2>
                <div class="box">{{ normalized_text if normalized_text else "提交文本后，这里会显示规范化结果。" }}</div>
                <div class="meta">
                    规范化步骤：去除特殊符号、全角转半角、繁体转简体{% if not opencc_available %}（当前未安装 OpenCC，繁简转换已跳过）{% endif %}
                </div>
            </section>

            <section class="panel">
                <h2>区块2：中文分词与词频统计</h2>
                <div class="box">{{ segmented_text if segmented_text else "提交文本后，这里会显示分词结果。" }}</div>
                <div class="meta">
                    最高频的 5 个词：
                    {% if top_words %}
                        {{ top_words | join("，") }}
                    {% else %}
                        暂无统计结果
                    {% endif %}
                </div>
                {% if jieba_available and freq_labels %}
                    <canvas id="freqChart" height="220"></canvas>
                {% elif not jieba_available %}
                    <div class="warning">未检测到 `jieba`，无法进行分词和词性标注。请先安装：`pip install jieba`。</div>
                {% endif %}
            </section>

            <section class="panel">
                <h2>区块3：词性标注结果</h2>
                <div class="legend">
                    名词 <span class="noun">红色</span>，动词 <span class="verb">蓝色</span>，形容词 <span class="adj">绿色</span>
                </div>
                <div class="box tags">
                    {% if pos_html %}
                        {{ pos_html }}
                    {% else %}
                        <span class="empty">提交文本后，这里会显示词性标注结果。</span>
                    {% endif %}
                </div>
            </section>
        </div>

        {% if jieba_error %}
            <div class="warning">`jieba` 导入失败：{{ jieba_error }}</div>
        {% endif %}
    </div>

    {% if jieba_available and freq_labels %}
    <script>
        const ctx = document.getElementById("freqChart");
        if (ctx) {
            new Chart(ctx, {
                type: "bar",
                data: {
                    labels: {{ freq_labels | tojson }},
                    datasets: [{
                        label: "词频",
                        data: {{ freq_values | tojson }},
                        backgroundColor: ["#2563eb", "#60a5fa", "#93c5fd", "#38bdf8", "#0ea5e9"],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { precision: 0 }
                        }
                    }
                }
            });
        }
    </script>
    {% endif %}
</body>
</html>
"""


def to_halfwidth(text: str) -> str:
    converted = []
    for char in text:
        code = ord(char)
        if code == 12288:
            converted.append(" ")
        elif 65281 <= code <= 65374:
            converted.append(chr(code - 65248))
        else:
            converted.append(char)
    return "".join(converted)


def to_simplified(text: str) -> str:
    if OPENCC_AVAILABLE and OPENCC is not None:
        return OPENCC.convert(text)
    return text


def normalize_text(text: str) -> str:
    text = to_halfwidth(text)
    text = to_simplified(text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_frequencies(words, limit: int = 5):
    counter = Counter(words)
    return counter.most_common(limit)


def pos_class(flag: str) -> str:
    if flag.startswith("n"):
        return "noun"
    if flag.startswith("v"):
        return "verb"
    if flag.startswith("a"):
        return "adj"
    return "other"


def build_pos_html(tagged_words) -> Markup:
    fragments = []
    for word, flag in tagged_words:
        css_class = pos_class(flag)
        fragments.append(
            f'<span class="tag {css_class}">{escape(word)} / {escape(flag)}</span>'
        )
    return Markup("".join(fragments))


@app.route("/", methods=["GET", "POST"])
def index():
    raw_text = ""
    normalized_text = ""
    segmented_text = ""
    top_words = []
    freq_labels = []
    freq_values = []
    pos_html = None

    if request.method == "POST":
        raw_text = request.form.get("text", "")
        normalized_text = normalize_text(raw_text)

        if JIEBA_AVAILABLE and normalized_text:
            words = [word.strip() for word in jieba.lcut(normalized_text) if word.strip()]
            segmented_text = " ".join(words)

            top_freq = get_top_frequencies(words, limit=5)
            top_words = [f"{word}({count})" for word, count in top_freq]
            freq_labels = [word for word, _ in top_freq]
            freq_values = [count for _, count in top_freq]

            tagged_words = [(item.word, item.flag) for item in pseg.cut(normalized_text) if item.word.strip()]
            pos_html = build_pos_html(tagged_words)

    return render_template_string(
        HTML_TEMPLATE,
        raw_text=raw_text,
        normalized_text=normalized_text,
        segmented_text=segmented_text,
        top_words=top_words,
        freq_labels=freq_labels,
        freq_values=freq_values,
        pos_html=pos_html,
        jieba_available=JIEBA_AVAILABLE,
        jieba_error=JIEBA_IMPORT_ERROR,
        opencc_available=OPENCC_AVAILABLE,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
