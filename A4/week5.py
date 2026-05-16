import streamlit as st
import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import spacy
from spacy import displacy
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置Matplotlib中文字体 ====================
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False              # 解决负号显示问题

# ==================== 下载必需的NLTK资源 ====================
@st.cache_resource
def download_nltk_resources():
    """下载NLTK所需的资源"""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # NLTK 3.8+ 需要 punkt_tab（word_tokenize 使用）
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

download_nltk_resources()

# ==================== 初始化BERT模型和Tokenizer ====================
@st.cache_resource
def load_bert_model():
    """加载BERT模型和Tokenizer（仅初始化一次）"""
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tokenizer, model

tokenizer, bert_model = load_bert_model()

# ==================== 加载spaCy模型 ====================
@st.cache_resource
def load_spacy_model():
    """加载spaCy英文模型（自动下载到临时目录）"""
    import sys
    import subprocess
    
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.warning("⚠️ spaCy模型初次加载，正在下载（约1-2分钟）...")
        try:
            # 使用--no-cache-dir避免权限问题
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--no-cache-dir"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                st.error(f"❌ 模型下载失败，请重新刷新页面")
                st.code(result.stderr, language="text")
                st.stop()
            
            nlp = spacy.load("en_core_web_sm")
            st.success("✅ spaCy模型已就绪！")
            return nlp
            
        except subprocess.TimeoutExpired:
            st.error("❌ 模型下载超时（>5分钟），请检查网络连接后刷新页面")
            st.stop()
        except Exception as e:
            st.error(f"❌ 下载过程出错: {str(e)}")
            st.stop()

nlp = load_spacy_model()

# ==================== 核心函数：Lesk算法 ====================
def wsd_lesk(sentence, target_word):
    """
    使用Lesk算法进行词义消歧
    
    Args:
        sentence (str): 输入的句子
        target_word (str): 目标多义词
    
    Returns:
        dict: 包含Synset和Definition的结果
    """
    tokens = word_tokenize(sentence.lower())
    
    # 查找目标词在句子中的位置
    target_word_lower = target_word.lower()
    if target_word_lower not in tokens:
        return {
            "error": f"目标词 '{target_word}' 未在句子中找到",
            "synset": None,
            "definition": None
        }
    
    # 使用Lesk算法获取词义
    synset = lesk(tokens, target_word_lower, pos='n')
    
    if synset is None:
        return {
            "error": f"未能为 '{target_word}' 找到合适的词义",
            "synset": None,
            "definition": None
        }
    
    return {
        "synset": str(synset),
        "definition": synset.definition(),
        "error": None
    }

# ==================== 核心函数：提取BERT上下文词向量 ====================
def get_contextual_embedding(sentence, target_word):
    """
    使用BERT提取目标词的上下文词向量
    
    Args:
        sentence (str): 输入的句子
        target_word (str): 目标词
    
    Returns:
        dict: 包含词向量和相关信息的结果
    """
    # Tokenize句子
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    
    # 获取BERT输出
    with torch.no_grad():
        outputs = bert_model(input_ids)
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
    
    # 查找目标词的位置
    target_tokens = tokenizer.tokenize(target_word.lower())
    target_word_lower = target_word.lower()
    
    # 在原始tokens中查找目标词
    target_indices = []
    for i, token in enumerate(tokens):
        if token.lower() == target_word_lower or token.lower().startswith(target_word_lower):
            target_indices.append(i)
    
    if not target_indices:
        return {
            "error": f"在BERT分词结果中未找到 '{target_word}'",
            "embedding": None,
            "tokens": tokens
        }
    
    # 获取目标词的平均词向量（如果被分成多个子词）
    target_embedding = last_hidden_state[0, target_indices].mean(dim=0).cpu().numpy()
    
    return {
        "embedding": target_embedding,
        "tokens": tokens,
        "target_indices": target_indices,
        "error": None
    }

# ==================== 核心函数：计算余弦相似度 ====================
def calculate_cosine_similarity(embedding1, embedding2):
    """
    计算两个词向量的余弦相似度
    
    Args:
        embedding1 (np.array): 第一个词向量
        embedding2 (np.array): 第二个词向量
    
    Returns:
        float: 余弦相似度分数（0-1之间）
    """
    if embedding1 is None or embedding2 is None:
        return None
    
    # 使用numpy计算余弦相似度
    # cosine_similarity = (A·B) / (||A|| * ||B||)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

# ==================== 核心函数：SRL启发式规则提取 ====================
def extract_srl_roles(sentence):
    """
    使用spaCy和启发式规则进行语义角色标注
    
    Args:
        sentence (str): 输入的句子
    
    Returns:
        dict: 包含提取的SRL角色和文档对象
    """
    doc = nlp(sentence)
    
    srl_roles = {
        'A0': None,  # 施事者（Agent）
        'Predicate': None,  # 谓词
        'A1': None,  # 受事者（Patient）
        'AM-LOC': None,  # 地点修饰语
        'AM-TMP': None   # 时间修饰语
    }
    
    # 寻找动词（谓词）
    verb_token = None
    for token in doc:
        if token.pos_ == 'VERB' or token.dep_ == 'ROOT':
            verb_token = token
            srl_roles['Predicate'] = (token.text, token.pos_, token.dep_)
            break
    
    # 如果未找到根动词，尝试找任意动词
    if verb_token is None:
        for token in doc:
            if token.pos_ == 'VERB':
                verb_token = token
                srl_roles['Predicate'] = (token.text, token.pos_, token.dep_)
                break
    
    if verb_token is None:
        return {'error': '未找到谓词（动词）', 'doc': doc, 'roles': srl_roles}
    
    # 遍历依存树找论元
    for token in doc:
        # A0: 主语（nsubj）
        if token.dep_ == 'nsubj' and srl_roles['A0'] is None:
            srl_roles['A0'] = (token.text, token.pos_, token.dep_)
        
        # A1: 直接宾语（dobj）
        elif token.dep_ == 'dobj' and srl_roles['A1'] is None:
            srl_roles['A1'] = (token.text, token.pos_, token.dep_)
        
        # AM-LOC: 地点修饰语（prep+pobj识别"in"等介词）
        elif token.dep_ == 'prep' and token.text.lower() in ['in', 'at', 'from', 'to']:
            for child in token.children:
                if child.dep_ == 'pobj' and srl_roles['AM-LOC'] is None:
                    srl_roles['AM-LOC'] = (f"{token.text} {child.text}", 'PREP+NOUN', 'prep+pobj')
        
        # AM-TMP: 时间修饰语
        elif token.dep_ == 'prep' and token.text.lower() in ['during', 'in', 'on']:
            for child in token.children:
                if child.dep_ == 'pobj':
                    # 检查是否是时间相关的词
                    time_keywords = ['year', 'month', 'week', 'day', 'time', 'period']
                    if any(kw in child.text.lower() for kw in time_keywords) and srl_roles['AM-TMP'] is None:
                        srl_roles['AM-TMP'] = (f"{token.text} {child.text}", 'PREP+NOUN', 'prep+pobj')
        
        # 时间副词修饰
        elif token.pos_ == 'NOUN' and any(kw in token.text.lower() for kw in ['year', 'time', 'day']):
            if srl_roles['AM-TMP'] is None:
                srl_roles['AM-TMP'] = (token.text, token.pos_, token.dep_)
    
    return {
        'error': None,
        'doc': doc,
        'roles': srl_roles
    }

# ==================== Streamlit应用主程序 ====================
def main():
    # 设置页面配置
    st.set_page_config(
        page_title="词义消歧与语义角色标注系统",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 标题
    st.title("🌐 词义消歧与语义角色标注系统")
    
    # 创建标签页
    tab1, tab2 = st.tabs(["WSD 词义消歧", "SRL 语义角色标注"])
    
    # ==================== 第一个标签页：WDS模块 ====================
    with tab1:
        st.header("词义消歧 (Word Sense Disambiguation)")
        st.markdown("---")
        
        # 第一阶段：用户输入第一个句子和目标词
        st.subheader("📝 第一阶段：输入句子和目标词")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentence1 = st.text_area(
                "输入第一个包含多义词的句子：",
                value="I went to the bank to deposit my money.",
                height=100,
                key="sentence1"
            )
        
        with col2:
            target_word = st.text_input(
                "输入目标多义词：",
                value="bank",
                key="target_word"
            )
        
        # 第一阶段分析按钮
        if st.button("🔍 第一阶段分析", key="btn_phase1"):
            if not sentence1.strip() or not target_word.strip():
                st.error("❌ 请输入句子和目标词")
            else:
                # 显示加载状态
                with st.spinner("正在分析中..."):
                    # Lesk算法分析
                    lesk_result = wsd_lesk(sentence1, target_word)
                    
                    # BERT上下文词向量提取
                    embedding_result1 = get_contextual_embedding(sentence1, target_word)
                    
                    # 保存到session_state以便后续使用（使用不同的键名避免冲突）
                    st.session_state.stored_sentence1 = sentence1
                    st.session_state.stored_target_word = target_word
                    st.session_state.stored_lesk_result = lesk_result
                    st.session_state.stored_embedding_result1 = embedding_result1
                    st.session_state.phase1_done = True
                
                st.success("✅ 第一阶段分析完成！详细结果见下方")
        
        # 显示第一阶段详细结果（如果已完成）
        if 'phase1_done' in st.session_state and st.session_state.phase1_done:
            with st.expander("📖 第一阶段详细结果 (点击展开)", expanded=True):
                
                # 显示Lesk算法结果
                st.markdown("### 📖 Lesk算法消歧结果")
                if st.session_state.stored_lesk_result.get("error"):
                    st.warning(f"⚠️ {st.session_state.stored_lesk_result['error']}")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**WordNet Synset:**\n`{st.session_state.stored_lesk_result['synset']}`")
                    with col2:
                        st.info(f"**英文定义:**\n{st.session_state.stored_lesk_result['definition']}")
                
                # 显示BERT上下文词向量信息
                st.markdown("### 🤖 BERT上下文词向量提取")
                if st.session_state.stored_embedding_result1.get("error"):
                    st.warning(f"⚠️ {st.session_state.stored_embedding_result1['error']}")
                else:
                    st.success(f"✅ 已成功提取词向量")
                    
                    # 获取完整的词向量
                    embedding_vector = st.session_state.stored_embedding_result1['embedding']
                    target_indices = st.session_state.stored_embedding_result1['target_indices']
                    
                    # 向量基本信息指标
                    st.markdown("#### 📊 向量基本信息")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("向量维度", len(embedding_vector))
                    with col2:
                        st.metric("目标词标记位置", str(target_indices))
                    with col3:
                        st.metric("向量范数", f"{np.linalg.norm(embedding_vector):.4f}")
                    
                    # 显示完整的上下文词向量
                    st.markdown("#### 🎯 目标词的上下文词向量（Contextual Embedding）")
                    st.info(f"""
                    **说明**：以下是目标词 **'{st.session_state.stored_target_word}'** 在句子中提取的BERT上下文词向量。
                    
                    这个768维的向量编码了该词在当前上下文中的完整语义信息。
                    """)
                    
                    # 展示向量值 - 分段显示
                    with st.expander("📋 查看完整词向量数值（768维）", expanded=False):
                        st.markdown("**所有维度的数值：**")
                        # 将向量分成多行显示
                        vec_str = ", ".join([f"{v:.6f}" for v in embedding_vector])
                        st.code(vec_str, language="")
                    
                    # 显示向量的不同部分
                    st.markdown("**向量分段展示：**")
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**前10维度（维度0-9）：**")
                        sample_1 = embedding_vector[:10]
                        st.write([f"{v:.6f}" for v in sample_1])
                    
                    with cols[1]:
                        st.markdown("**中间部分（维度379-388）：**")
                        sample_2 = embedding_vector[379:388]
                        st.write([f"{v:.6f}" for v in sample_2])
                    
                    with cols[2]:
                        st.markdown("**后10维度（维度758-767）：**")
                        sample_3 = embedding_vector[-10:]
                        st.write([f"{v:.6f}" for v in sample_3])
                    
                    # 向量统计信息
                    st.markdown("#### 📈 向量统计信息")
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    with stats_col1:
                        st.metric("均值", f"{np.mean(embedding_vector):.6f}")
                    with stats_col2:
                        st.metric("标准差", f"{np.std(embedding_vector):.6f}")
                    with stats_col3:
                        st.metric("最大值", f"{np.max(embedding_vector):.6f}")
                    with stats_col4:
                        st.metric("最小值", f"{np.min(embedding_vector):.6f}")
                    
                    # 可视化词向量分布
                    st.markdown("#### 📊 向量数值分布直方图")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(embedding_vector, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
                    ax.set_xlabel('向量数值')
                    ax.set_ylabel('频率')
                    ax.set_title(f'目标词 "{st.session_state.stored_target_word}" 的BERT词向量分布')
                    st.pyplot(fig)
        
        st.markdown("---")
        
        # 第二阶段：对比验证（需要先完成第一阶段）
        st.subheader("🔄 第二阶段：对比验证 - 余弦相似度")
        
        if 'phase1_done' not in st.session_state or not st.session_state.phase1_done:
            st.info("💡 请先完成第一阶段的分析")
        else:
            st.markdown(f"**当前目标词：** `{st.session_state.stored_target_word}`")
            st.markdown(f"**第一句：** {st.session_state.stored_sentence1}")
            
            # 输入第二个句子
            sentence2 = st.text_area(
                "输入第二个包含该目标词的句子：",
                value="I sat by the river bank.",
                height=100,
                key="sentence2"
            )
            
            # 第二阶段分析按钮
            if st.button("🔗 计算余弦相似度", key="btn_phase2"):
                if not sentence2.strip():
                    st.error("❌ 请输入第二个句子")
                else:
                    with st.spinner("计算中..."):
                        # 提取第二个句子的词向量
                        embedding_result2 = get_contextual_embedding(sentence2, st.session_state.stored_target_word)
                        
                        if embedding_result2.get("error"):
                            st.warning(f"⚠️ {embedding_result2['error']}")
                        elif st.session_state.stored_embedding_result1.get("error"):
                            st.warning(f"⚠️ 第一阶段词向量获取失败")
                        else:
                            # 计算余弦相似度
                            similarity = calculate_cosine_similarity(
                                st.session_state.stored_embedding_result1['embedding'],
                                embedding_result2['embedding']
                            )
                            
                            # 显示结果
                            st.markdown("### 📊 余弦相似度分析结果")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("相似度分数", f"{similarity:.4f}")
                            with col2:
                                similarity_percent = similarity * 100
                                st.metric("相似度百分比", f"{similarity_percent:.2f}%")
                            with col3:
                                # 语义判断
                                if similarity > 0.7:
                                    interpretation = "✅ 高度相似"
                                elif similarity > 0.5:
                                    interpretation = "⚠️ 中度相似"
                                else:
                                    interpretation = "❌ 低度相似"
                                st.metric("语义判断", interpretation)
                            
                            st.markdown("---")
                            
                            # 详细对比
                            st.markdown("### 📋 对比详情")
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                st.markdown("**第一句的词向量统计：**")
                                emb1 = st.session_state.stored_embedding_result1['embedding']
                                st.write({
                                    "维度": len(emb1),
                                    "均值": f"{np.mean(emb1):.6f}",
                                    "标准差": f"{np.std(emb1):.6f}",
                                    "最大值": f"{np.max(emb1):.6f}",
                                    "最小值": f"{np.min(emb1):.6f}"
                                })
                            
                            with detail_col2:
                                st.markdown("**第二句的词向量统计：**")
                                emb2 = embedding_result2['embedding']
                                st.write({
                                    "维度": len(emb2),
                                    "均值": f"{np.mean(emb2):.6f}",
                                    "标准差": f"{np.std(emb2):.6f}",
                                    "最大值": f"{np.max(emb2):.6f}",
                                    "最小值": f"{np.min(emb2):.6f}"
                                })
                            
                            # 解释
                            st.markdown("### 💡 分析解释")
                            st.markdown(f"""
                            - **高相似度 (0.7-1.0)**：表示目标词在两个句子中的语义用法非常相似，上下文环境接近
                            - **中相似度 (0.5-0.7)**：表示目标词有相似的语义特征，但上下文有一定差异
                            - **低相似度 (0.0-0.5)**：表示目标词在两个句子中的用法差异较大，可能代表不同的词义
                            
                            当前结果显示相似度为 **{similarity:.4f}**，说明 "{st.session_state.stored_target_word}" 在这两个句子中的上下文差异为 **{(1-similarity)*100:.2f}%**。
                            """)
    
    # ==================== 第二个标签页：SRL模块 ====================
    with tab2:
        st.header("语义角色标注 (Semantic Role Labeling)")
        st.markdown("---")
        
        # SRL输入
        st.subheader("📝 输入句子")
        srl_sentence = st.text_area(
            "输入要进行语义角色标注的句子：",
            value="Apple is manufacturing new smartphones in China this year.",
            height=100,
            key="srl_sentence"
        )
        
        # SRL分析按钮
        if st.button("🔍 执行语义角色标注", key="btn_srl_analyze"):
            if not srl_sentence.strip():
                st.error("❌ 请输入句子")
            else:
                with st.spinner("正在分析中..."):
                    # 执行SRL提取
                    srl_result = extract_srl_roles(srl_sentence)
                    
                    # 保存到session_state（使用不同的键名避免冲突）
                    st.session_state.stored_srl_sentence = srl_sentence
                    st.session_state.stored_srl_result = srl_result
                    st.session_state.srl_analysis_done = True
                
                # 分析结果显示逻辑（按钮点击时直接显示）
                if srl_result.get('error'):
                    st.warning(f"⚠️ {srl_result['error']}")
                else:
                    st.markdown("### 📊 语义角色提取结果")
                    
                    # 构建结果表格
                    roles = srl_result['roles']
                    
                    table_data = []
                    role_names = ['A0 (施事者)', 'Predicate (谓词)', 'A1 (受事者)', 'AM-LOC (地点)', 'AM-TMP (时间)']
                    role_keys = ['A0', 'Predicate', 'A1', 'AM-LOC', 'AM-TMP']
                    
                    for role_name, role_key in zip(role_names, role_keys):
                        if roles[role_key] is not None:
                            text, pos, dep = roles[role_key]
                            table_data.append({
                                '语义角色': role_name,
                                '提取文本': text,
                                '词性标签': pos,
                                '依存关系': dep
                            })
                        else:
                            table_data.append({
                                '语义角色': role_name,
                                '提取文本': '-',
                                '词性标签': '-',
                                '依存关系': '-'
                            })
                    
                    # 显示表格
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    
                    # 显示依存关系图
                    st.markdown("### 📈 依存关系图")
                    st.markdown("以下是句子的依存句法树，帮助理解SRL提取的依据：")
                    
                    try:
                        doc = srl_result['doc']
                        # 生成displacy依存图
                        html_dep = displacy.render(doc, style="dep", manual=False, page=True, options={"compact": False})
                        # 使用unsafe_allow_html来渲染HTML
                        st.write(html_dep, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"⚠️ 无法生成依存图：{str(e)}")
                    
                    st.markdown("---")
                    
                    # 显示详细的分析说明
                    st.markdown("### 💡 SRL分析说明")
                    st.markdown("""
                    **提取规则：**
                    - **A0 (施事者)**：通过nsubj（名词性主语）依存关系识别
                    - **Predicate (谓词)**：通过ROOT或VERB标签识别主要动词
                    - **A1 (受事者)**：通过dobj（直接宾语）依存关系识别
                    - **AM-LOC (地点修饰语)**：识别介词短语（如"in China"），介词包括：in, at, from, to等
                    - **AM-TMP (时间修饰语)**：识别时间相关的表述（如"this year"），关键词包括：year, month, week, day, time等
                    
                    **依存图说明：**
                    - 箭头指向依存关系的从属对象
                    - 标签显示依存关系类型（nsubj, dobj, prep等）
                    - ROOT指向句子的中心谓词
                    """)
                    
                    # 最后显示完成提示
                    st.success("✅ SRL分析完成！")


if __name__ == "__main__":
    main()
