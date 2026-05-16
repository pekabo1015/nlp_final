import streamlit as st
import subprocess
import sys

# 中文注释：尽量让用户在本地一键运行。运行时会自动下载模型（如果缺失）。

# 启动时即导入 benepar，便于 spaCy 注册管道；未安装时后续会走“未检测到”提示
try:
    import benepar
except ImportError:
    benepar = None


def ensure_spacy_model(model_name='en_core_web_sm'):
    """确保 spaCy 模型已安装。"""
    try:
        import spacy
        spacy.load(model_name)
        return True
    except Exception:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', model_name], check=False)
        try:
            spacy.load(model_name)
            return True
        except Exception:
            return False


def ensure_benepar_model(model_name='benepar_en3'):
    """确保 benepar 模型已安装。"""
    try:
        import benepar
        benepar.download(model_name)
        return True
    except Exception:
        return False


@st.cache_resource
def load_nlp():
    """加载 spaCy 英文模型和 benepar。先尝试独立解析器（不依赖管道注册），再尝试加入 spaCy 管道。"""
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        ensure_spacy_model('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    has_benepar = False
    standalone_parser = None
    benepar_error = None

    if benepar is None:
        benepar_error = "未安装 benepar，请运行: pip install benepar"
        return nlp, has_benepar, standalone_parser, benepar_error

    # 1) 先确保模型已下载
    ensure_benepar_model('benepar_en3')

    # 2) 优先尝试独立解析器（不依赖 spaCy 管道注册与 config 校验）
    try:
        from benepar.integrations.downloader import load_trained_model
        standalone_parser = load_trained_model('benepar_en3')
        has_benepar = True
        return nlp, has_benepar, standalone_parser, None
    except Exception as e:
        benepar_error = str(e)

    # 3) 独立解析器失败时，再尝试加入 spaCy 管道（validate=False 避免 REGEX 等配置错误）
    try:
        if 'benepar' not in nlp.pipe_names:
            nlp.add_pipe('benepar', config={'model': 'benepar_en3'}, validate=False)
        has_benepar = True
        benepar_error = None
    except Exception as e:
        benepar_error = (benepar_error or "") + "; 管道添加失败: " + str(e)

    return nlp, has_benepar, standalone_parser, benepar_error


def render_dependency_svg(doc):
    """使用 displaCy 生成依存分析 SVG。"""
    from spacy import displacy
    svg = displacy.render(doc, style='dep', jupyter=False)
    return svg


def get_constituency_text(doc, has_benepar, standalone_parser=None, benepar_error=None):
    """获取文本形式成分树。若使用独立解析器，则用 standalone_parser 解析首句。"""
    if not has_benepar:
        msg = '未检测到 benepar 成分解析器，需先安装 benepar。'
        if benepar_error:
            msg += '\n\n若已安装，可参考以下错误信息排查：\n' + benepar_error
            if 'build_inputs_with_special_tokens' in (benepar_error or ''):
                msg += '\n\n【建议】多为 transformers 版本过新导致，可在虚拟环境中执行：\npip install "transformers>=4.25,<4.44"'
        return msg
    sent_list = list(doc.sents)
    if not sent_list:
        return '无法检测到句子。'
    try:
        if standalone_parser is not None:
            # 独立解析器模式：用首句解析，return_compressed=False 得到 nltk.Tree 再转字符串
            from benepar.integrations.spacy_plugin import SentenceWrapper
            batch = [SentenceWrapper(sent_list[0])]
            trees = list(standalone_parser.parse(batch, return_compressed=False))
            if not trees:
                return '成分句法解析失败：未得到树。'
            tree = trees[0]
            return tree.pformat() if hasattr(tree, 'pformat') else str(tree)
        # 管道模式：从 span 扩展取括号树
        return sent_list[0]._.parse_string
    except Exception as e:
        return f'成分句法解析失败：{e}'


def detect_ambiguity(doc):
    """简单结构歧义提示。"""
    indicators = []
    text = doc.text.lower()
    if 'with' in text:
        indicators.append('注意：含有 with 的句子可能发生介词短语附着歧义。')
    if 'that' in text:
        indicators.append('注意：that 可能引入限定/非限定歧义。')
    if len(list(doc.sents)) > 1:
        indicators.append('含多句子，句子边界可能引发歧义。')
    return indicators or ['未检测到显式歧义提示，仍需人工审阅。']


def extract_core_arguments(doc):
    """从依存解析中提取核心论元：nsubj, dobj, pobj, ROOT。"""
    rows = []
    for tok in doc:
        dep = tok.dep_.lower()
        if dep in {'nsubj', 'dobj', 'pobj'} or dep == 'root':
            rows.append({
                '论元类型': 'ROOT' if dep == 'root' else tok.dep_,
                '词': tok.text,
                '词性': tok.pos_,
                '父节点': tok.head.text,
                '依存关系': tok.dep_
            })
    return rows


def main():
    st.set_page_config(page_title='成分 + 依存句法树可视化', layout='wide')

    # 顶部标题
    st.markdown('# 句法树可视化平台\n---')
    st.markdown('这是一个演示句子依存句法和成分句法的 Streamlit 小应用。')

    nlp, has_benepar, standalone_parser, benepar_error = load_nlp()

    # 输入框
    sentence = st.text_input('请输入英文句子：', 'The boy saw the man with the telescope.')

    # 如果输入为空，提示
    if not sentence:
        st.warning('请在输入框中填入一句英文句子。')
        return

    doc = nlp(sentence)
    dep_svg = render_dependency_svg(doc)
    constituency_text = get_constituency_text(doc, has_benepar, standalone_parser, benepar_error)
    ambiguity_hints = detect_ambiguity(doc)
    core_args = extract_core_arguments(doc)

    st.markdown('### :left_speech_bubble: 依存句法树（Dependency）')
    st.markdown('> 由 spaCy displaCy 生成的依存关系图。')
    st.markdown('---')
    st.markdown('#### 依存图（SVG）')
    st.components.v1.html(dep_svg, height=420)
    st.markdown('#### 依存关系（文本）')
    st.code('\n'.join([f'{tok.text:<12} -> {tok.head.text:<12} ({tok.dep_})' for tok in doc]), language='text')

    st.markdown('---')
    st.markdown('### :books: 成分句法树（Constituency）')
    st.markdown('> 使用 benepar 生成成分解析树。如果未安装 benepar，会显示文本提示。')
    st.markdown('---')
    st.markdown('#### 成分树（括号文本）')
    st.code(constituency_text, language='text')

    st.markdown('---')
    st.markdown('### 🧠 结构歧义提示')
    for hint in ambiguity_hints:
        st.info(hint)

    st.markdown('---')
    st.markdown('## 🔍 核心论元提取器')
    st.markdown('提取依存句法中的核心论元：nsubj、dobj、pobj、ROOT。')
    if core_args:
        st.dataframe(core_args)
    else:
        st.warning('未检测到 nsubj、dobj、pobj 或 ROOT 论元（输入句子过短或解析异常）。')


if __name__ == '__main__':
    main()

