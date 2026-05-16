# main_updated：多算法分词对比与统计可视化

import re
from collections import Counter
from html import escape

try:
    from flask import Flask, render_template_string, request
except ImportError:
    Flask = None
    render_template_string = None
    request = None

try:
    from markupsafe import Markup
except ImportError:
    Markup = None

# ---------------------------------------------------------------------------
# 分词库：jieba（必选）+ snownlp（可选，用于算法对比）
# ---------------------------------------------------------------------------
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

try:
    from snownlp import SnowNLP
    SNOWNLP_AVAILABLE = True
except ImportError:
    SNOWNLP_AVAILABLE = False

# ---------------------------------------------------------------------------
# 多算法分词：返回列表 [(算法名, 词列表), ...]，便于歧义句对比
# ---------------------------------------------------------------------------
def segment_jieba_default(text: str):
    if not JIEBA_AVAILABLE:
        return []
    return [w.strip() for w in jieba.lcut(text) if w.strip()]

def segment_jieba_full(text: str):
    if not JIEBA_AVAILABLE:
        return []
    return [w.strip() for w in jieba.lcut(text, cut_all=True) if w.strip()]

def segment_jieba_search(text: str):
    if not JIEBA_AVAILABLE:
        return []
    return [w.strip() for w in jieba.lcut_for_search(text) if w.strip()]

def segment_snownlp(text: str):
    if not SNOWNLP_AVAILABLE:
        return []
    try:
        return [w.strip() for w in SnowNLP(text).words if w.strip()]
    except Exception:
        return []

# 算法注册表： (显示名, 函数)，共四种分词算法
SEGMENTERS = [
    ("jieba 精确模式", segment_jieba_default),
    ("jieba 全模式", segment_jieba_full),
    ("jieba 搜索引擎模式", segment_jieba_search),
    ("SnowNLP", segment_snownlp),
]

def run_all_segmenters(text: str):
    """对文本运行所有可用分词器，返回 [(算法名, 词列表), ...]"""
    results = []
    for name, func in SEGMENTERS:
        try:
            words = func(text)
            if words is not None:
                results.append((name, words))
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# 统计分析：词数、平均词长、唯一词数；两两 Jaccard 相似度
# ---------------------------------------------------------------------------
def stat_segments(words: list):
    if not words:
        return {"count": 0, "avg_len": 0.0, "unique": 0}
    lengths = [len(w) for w in words]
    return {
        "count": len(words),
        "avg_len": round(sum(lengths) / len(lengths), 2),
        "unique": len(set(words)),
    }

def jaccard_similarity(words_a: list, words_b: list):
    """词集合的 Jaccard 相似度"""
    set_a, set_b = set(words_a), set(words_b)
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return round(inter / union, 4) if union else 0.0

def build_similarity_matrix(algo_results: list):
    """algo_results: [(name, words), ...]，返回 (names, 二维相似度矩阵)"""
    names = [r[0] for r in algo_results]
    words_list = [r[1] for r in algo_results]
    n = len(names)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = jaccard_similarity(words_list[i], words_list[j])
    return names, matrix


# ---------------------------------------------------------------------------
# 文本规范化（与原 main.py 一致）
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 词性标注（沿用 jieba）
# ---------------------------------------------------------------------------
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

def build_pos_html(tagged_words):
    fragments = []
    for word, flag in tagged_words:
        css_class = pos_class(flag)
        fragments.append(
            f'<span class="tag {css_class}">{escape(word)} / {escape(flag)}</span>'
        )
    html_text = "".join(fragments)
    if Markup is None:
        return html_text
    return Markup(html_text)


# ---------------------------------------------------------------------------
# Flask 应用与模板
# ---------------------------------------------------------------------------
if Flask is not None:
    app = Flask(__name__)
else:
    app = None

HTML_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>中文分词与词性标注 · 多算法对比</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #f0f4f8;
            --card-bg: #ffffff;
            --card-soft: rgba(255,255,255,0.92);
            --shadow: 0 4px 20px rgba(100,116,139,0.06);
            --shadow-lg: 0 8px 32px rgba(100,116,139,0.1);
            --radius: 18px;
            --radius-lg: 22px;
            --radius-sm: 12px;
            --text: #334155;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --accent-soft: #dbeafe;
            --progress-bg: #e2e8f0;
            --progress-fill: #93c5fd;
        }
        body { margin: 0; font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
        .container { max-width: 1280px; margin: 0 auto; padding: 28px; }
        .title-lg { margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 600; color: var(--text); letter-spacing: -0.02em; }
        .desc { margin-bottom: 22px; color: var(--text-muted); font-size: 0.95rem; }

        .input-card {
            background: var(--card-soft);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 22px;
            margin-bottom: 24px;
        }
        .input-card label { display: block; font-weight: 600; color: var(--text); margin-bottom: 10px; }
        textarea {
            width: 100%; min-height: 160px; resize: vertical;
            border: none; border-radius: var(--radius-sm); padding: 16px;
            box-sizing: border-box; font-size: 15px; line-height: 1.7;
            background: #f8fafc; color: var(--text);
        }
        textarea::placeholder { color: #94a3b8; }
        textarea:focus { outline: none; box-shadow: 0 0 0 2px var(--accent-soft); }
        button {
            margin-top: 14px; background: var(--accent); color: #fff; border: none;
            padding: 12px 22px; border-radius: var(--radius-sm); cursor: pointer; font-size: 15px;
            box-shadow: 0 2px 10px rgba(59,130,246,0.25);
        }
        button:hover { filter: brightness(1.05); }

        .top-row-card, .bottom-row-card {
            background: var(--card-soft);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            padding: 24px;
            border: none;
            margin-bottom: 26px;
        }
        .card-caption {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 18px;
            letter-spacing: 0.02em;
            padding-left: 12px;
            border-left: 4px solid var(--accent);
        }
        .top-row-inner {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) { .top-row-inner { grid-template-columns: 1fr; } }
        .bottom-row-inner { display: flex; flex-direction: column; gap: 22px; }
        .block {
            min-width: 0;
            background: var(--card-bg);
            border-radius: var(--radius-sm);
            box-shadow: var(--shadow);
            padding: 18px 26px;
            border: none;
        }
        .block .title-sm {
            margin: 0 0 12px 0;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-muted);
        }
        .block .box { min-height: 70px; }
        .block .meta { margin-top: 10px; }

        .card {
            background: var(--card-soft);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 20px;
            border: none;
        }
        .card h2 { margin: 0 0 14px 0; font-size: 1.1rem; font-weight: 600; color: var(--text); }
        .box {
            min-height: 80px; background: #f8fafc;
            border-radius: var(--radius-sm); padding: 16px; line-height: 1.85;
            white-space: pre-wrap; word-break: break-word;
            border: none;
        }
        .meta { margin-top: 12px; color: var(--text-muted); font-size: 13px; }
        .tags { margin-top: 8px; line-height: 2; }
        .tag {
            display: inline-block; margin: 0 8px 8px 0; padding: 4px 10px;
            border-radius: 999px; background: var(--accent-soft); font-size: 13px;
        }
        .noun { color: #dc2626; font-weight: 600; }
        .verb { color: #2563eb; font-weight: 600; }
        .adj { color: #16a34a; font-weight: 600; }
        .other { color: var(--text-muted); }
        .legend { margin-bottom: 10px; font-size: 13px; color: var(--text-muted); }
        .warning {
            margin-top: 12px; padding: 12px 14px; background: #fffbeb;
            border-radius: var(--radius-sm); color: #b45309; font-size: 14px;
            border: none; box-shadow: none;
        }
        .empty { color: #94a3b8; }

        .progress-wrap { margin-top: 8px; }
        .progress-label { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; display: flex; justify-content: space-between; }
        .progress-bar {
            height: 8px; border-radius: 999px; background: var(--progress-bg); overflow: hidden;
        }
        .progress-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #93c5fd, #60a5fa); transition: width 0.3s ease; }

        .stats-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 12px; }
        .stats-table th, .stats-table td { padding: 10px 14px; text-align: left; border: none; }
        .stats-table th { background: #f1f5f9; color: var(--text-muted); font-weight: 500; border-radius: var(--radius-sm) var(--radius-sm) 0 0; }
        .stats-table tr:not(:last-child) td { border-bottom: 1px solid #f1f5f9; }
        .stats-table td { background: var(--card-bg); }
        .stats-table tbody tr:hover td { background: #f8fafc; }
        .stat-row-bar { margin-top: 6px; }
        .stat-row-bar .progress-bar { max-width: 120px; }

        .algo-grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
        .algo-card {
            background: var(--card-bg);
            border-radius: var(--radius-sm);
            box-shadow: var(--shadow);
            padding: 14px;
            border: none;
        }
        .algo-card .title-sm { margin-bottom: 8px; font-size: 0.9rem; }
        .algo-card .box { min-height: 50px; }

        .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; align-items: stretch; }
        @media (max-width: 700px) { .charts-row { grid-template-columns: 1fr; } }
        .chart-block { height: 280px; display: flex; flex-direction: column; min-height: 280px; }
        .chart-block .meta { margin-bottom: 8px; flex-shrink: 0; }
        .chart-block .chart-body, .chart-block .heatmap-wrap { flex: 1; min-height: 0; overflow: auto; }
        .chart-block .chart-body { position: relative; border-radius: var(--radius-sm); }
        .chart-block .chart-body canvas { position: absolute; left: 0; top: 0; width: 100% !important; height: 100% !important; border-radius: var(--radius-sm); }
        canvas { border-radius: var(--radius-sm); }
        .heatmap-wrap { margin-top: 0; border-radius: var(--radius-sm); background: #f8fafc; padding: 8px; }
        .heatmap { border-collapse: collapse; font-size: 12px; border: none; }
        .heatmap th, .heatmap td { padding: 8px 10px; min-width: 44px; text-align: center; border: none; }
        .heatmap th { background: #f1f5f9; color: var(--text-muted); font-weight: 500; }
        .heatmap td { background: #fff; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title-lg">中文文本处理 · 多算法分词对比</h1>
        <div class="desc">文本规范化、多种分词算法对比、统计分析与可视化、词性标注。</div>

        <form method="post" class="input-card">
            <label for="text">请输入中文长文本（含歧义句更易看出算法差异）</label>
            <textarea id="text" name="text" placeholder="例如：结婚的和尚未结婚的、南京市长江大桥、今天天气很好我们正在开发自然语言处理应用。">{{ raw_text }}</textarea>
            <button type="submit">开始处理</button>
        </form>

        <section class="top-row-card">
            <div class="card-caption">标准 jieba 分词结果</div>
            <div class="top-row-inner">
                <div class="block">
                    <h3 class="title-sm">区块1：文本规范化</h3>
                    <div class="box">{{ normalized_text if normalized_text else "提交文本后这里显示规范化结果。" }}</div>
                    <div class="meta">去除特殊符号、全角转半角、繁体转简体{% if not opencc_available %}（未安装 OpenCC 则跳过繁简）{% endif %}</div>
                </div>
                <div class="block">
                    <h3 class="title-sm">区块2：默认分词与词频</h3>
                    <div class="box">{{ segmented_text if segmented_text else "提交后显示分词与词频。" }}</div>
                    <div class="meta">最高频 5 词：{% if top_words %}{{ top_words | join("，") }}{% else %}暂无{% endif %}</div>
                    {% if jieba_available and freq_labels %}
                    <div class="progress-wrap">
                        {% for label in freq_labels %}
                        <div class="progress-label"><span>{{ label }}</span><span>{{ freq_values[loop.index0] }}</span></div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {{ (freq_values[loop.index0] / max_freq * 100) if max_freq else 0 }}%;"></div></div>
                        {% endfor %}
                    </div>
                    <canvas id="freqChart" height="200"></canvas>
                    {% elif not jieba_available %}
                    <div class="warning">未安装 jieba，请：pip install jieba</div>
                    {% endif %}
                </div>
                <div class="block">
                    <h3 class="title-sm">区块3：词性标注</h3>
                    <div class="legend">名词 <span class="noun">红</span>，动词 <span class="verb">蓝</span>，形容词 <span class="adj">绿</span></div>
                    <div class="box tags">{% if pos_html %}{{ pos_html }}{% else %}<span class="empty">提交后显示词性标注。</span>{% endif %}</div>
                </div>
            </div>
        </section>

        <section class="bottom-row-card">
            <div class="card-caption">多算法对比与统计分析</div>
            <div class="bottom-row-inner">
                <div class="block">
                    <h3 class="title-sm">区块4：多算法分词结果对比</h3>
                    {% if algo_results %}
                    <div class="algo-grid">
                        {% for name, words in algo_results %}
                        <div class="algo-card">
                            <h3 class="title-sm">{{ name }}</h3>
                            <div class="box">{{ words | join(" ") or "（无结果）" }}</div>
                        </div>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="box"><span class="empty">提交文本后此处显示各算法分词结果。</span></div>
                    {% endif %}
                </div>
                <div class="block">
                    <h3 class="title-sm">区块5：统计分析与可视化</h3>
                    {% if algo_results and stats_rows %}
                    <p class="meta">各算法：词数、平均词长、唯一词数；算法间 Jaccard 相似度。</p>
                    <table class="stats-table">
                        <thead><tr><th>算法</th><th>词数</th><th>平均词长</th><th>唯一词数</th></tr></thead>
                        <tbody>
                            {% for row in stats_rows %}
                            <tr>
                                <td>{{ row.name }}</td><td>{{ row.count }}</td><td>{{ row.avg_len }}</td><td>{{ row.unique }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% if max_count %}
                    <div class="progress-wrap stat-row-bar" style="margin-top:14px;">
                        {% for row in stats_rows %}
                        <div class="progress-label"><span>{{ row.name }}</span><span>{{ row.count }} 词</span></div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {{ (row.count / max_count * 100) if max_count else 0 }}%;"></div></div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    <div class="charts-row">
                        <div class="chart-block">
                            <p class="meta">各算法分词数量对比</p>
                            <div class="chart-body"><canvas id="countChart"></canvas></div>
                        </div>
                        <div class="chart-block">
                            <p class="meta">算法间 Jaccard 相似度</p>
                            <div class="heatmap-wrap">
                                <table class="heatmap">
                                    <thead><tr><th></th>{% for n in sim_names %}<th>{{ n }}</th>{% endfor %}</tr></thead>
                                    <tbody>
                                        {% for row in sim_matrix %}
                                        <tr><th>{{ sim_names[loop.index0] }}</th>{% for cell in row %}<td style="background: rgba(59,130,246,{{ cell * 0.6 + 0.2 }}); color: {{ '#fff' if cell > 0.5 else '#334155' }};">{{ "%.2f" | format(cell) }}</td>{% endfor %}</tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    {% else %}
                    <div class="box"><span class="empty">提交文本后此处显示统计表与图表。</span></div>
                    {% endif %}
                </div>
            </div>
        </section>

        {% if jieba_error %}
        <div class="warning" style="margin-top:20px;">jieba 导入失败：{{ jieba_error }}</div>
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
                    datasets: [{ label: "词频", data: {{ freq_values | tojson }},
                        backgroundColor: ["#2563eb","#60a5fa","#93c5fd","#38bdf8","#0ea5e9"], borderRadius: 8 }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { precision: 0 } },
                        x: { grid: { display: false } }
                    },
                    datasets: { bar: { barPercentage: 0.55, categoryPercentage: 0.8 } }
                }
            });
        }
    </script>
    {% endif %}

    {% if algo_results and stats_rows and count_labels is defined and count_values is defined %}
    <script>
        (function(){
            const countCtx = document.getElementById("countChart");
            if (countCtx) {
                new Chart(countCtx, {
                    type: "bar",
                    data: {
                        labels: {{ count_labels | tojson }},
                        datasets: [{ label: "词数", data: {{ count_values | tojson }},
                            backgroundColor: "#2563eb", borderRadius: 8 }]
                    },
                    options: {
                        indexAxis: "y",
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { beginAtZero: true, ticks: { precision: 0 } },
                            y: { grid: { display: false } }
                        },
                        datasets: { bar: { barPercentage: 0.55, categoryPercentage: 0.8 } }
                    }
                });
            }
        })();
    </script>
    {% endif %}
</body>
</html>
"""


if app is not None:
    @app.route("/", methods=["GET", "POST"])
    def index():
        raw_text = ""
        normalized_text = ""
        segmented_text = ""
        top_words = []
        freq_labels = []
        freq_values = []
        pos_html = None
        algo_results = []
        stats_rows = []
        sim_names = []
        sim_matrix = []
        count_labels = []
        count_values = []
        max_freq = 1
        max_count = 1

        if request.method == "POST":
            raw_text = request.form.get("text", "")
            normalized_text = normalize_text(raw_text)

            if normalized_text:
                if JIEBA_AVAILABLE:
                    words = segment_jieba_default(normalized_text)
                    segmented_text = " ".join(words)
                    top_freq = get_top_frequencies(words, limit=5)
                    top_words = [f"{w}({c})" for w, c in top_freq]
                    freq_labels = [w for w, _ in top_freq]
                    freq_values = [c for _, c in top_freq]
                    max_freq = max(freq_values, default=1)
                    tagged = [
                        (x.word, x.flag)
                        for x in pseg.cut(normalized_text)
                        if x.word.strip()
                    ]
                    pos_html = build_pos_html(tagged)

                algo_results = run_all_segmenters(normalized_text)

                if algo_results:
                    stats_rows = []
                    for name, words in algo_results:
                        s = stat_segments(words)
                        stats_rows.append(
                            {
                                "name": name,
                                "count": s["count"],
                                "avg_len": s["avg_len"],
                                "unique": s["unique"],
                            }
                        )
                    count_labels = [r["name"] for r in stats_rows]
                    count_values = [r["count"] for r in stats_rows]
                    max_count = max((r["count"] for r in stats_rows), default=1)

                    sim_names, sim_matrix = build_similarity_matrix(algo_results)

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
            algo_results=algo_results,
            stats_rows=stats_rows,
            sim_names=sim_names,
            sim_matrix=sim_matrix,
            count_labels=count_labels,
            count_values=count_values,
            max_freq=max_freq,
            max_count=max_count,
        )

    if __name__ == "__main__":
        app.run(host="127.0.0.1", port=5001, debug=True)
