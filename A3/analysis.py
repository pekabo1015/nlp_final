import os
import urllib.request
import zipfile

# NLTK 3.9+ 在「import nltk」时会立刻加载 stem/wordnet，此时若磁盘上还没有 wordnet 会直接 LookupError，
# 因此必须在 import nltk 之前把语料放到 nltk_data（不能用 nltk.download，那时还导不进 nltk）。
def _ensure_nltk_wordnet():
    nltk_data = os.path.join(os.path.expanduser('~'), 'nltk_data')
    corpora = os.path.join(nltk_data, 'corpora')
    wordnet_dir = os.path.join(corpora, 'wordnet')
    if os.path.isdir(wordnet_dir) and os.listdir(wordnet_dir):
        return
    os.makedirs(corpora, exist_ok=True)
    url = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'
    zip_path = os.path.join(corpora, 'wordnet.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(corpora)


_ensure_nltk_wordnet()

import streamlit as st
import pandas as pd
# Missing scikit-learn in Python environment can cause ModuleNotFoundError: No module named 'sklearn'
# resolve with: pip install scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
import numpy as np

# punkt / punkt_tab：分句分词（在成功 import nltk 之后再下载即可）
for data_path, download_name in [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab/english', 'punkt_tab'),
]:
    try:
        nltk.data.find(data_path)
    except LookupError:
        nltk.download(download_name, quiet=True)

st.set_page_config(page_title='语义分析综合测试平台', layout='wide')
st.title('语义分析综合测试平台')

tabs = st.tabs(['传统统计文本表示', '语义词向量', '主题建模', '文本分类/评估'])

with tabs[0]:
    st.header('模块 1：传统统计文本表示（TF-IDF + LSA）')

    raw_text = st.text_area('输入英文语料（可多句）', height=220, placeholder='输入英文文本，例如：\nThis is the first sentence.\nHere is another sentence.')

    if raw_text:
        sentences = [s.strip() for s in nltk.sent_tokenize(raw_text) if s.strip()]

        if len(sentences) == 0:
            st.warning('请提供至少一句有效英文语料。')
        else:
            st.subheader(f'分句结果：共 {len(sentences)} 个句子')
            for i, sent in enumerate(sentences, 1):
                st.markdown(f'{i}. {sent}')

            matrix_type = st.radio('选择表示方式', ['TF-IDF', 'One-hot (CountVectorizer)'])

            if matrix_type == 'TF-IDF':
                vectorizer = TfidfVectorizer()
            else:
                vectorizer = CountVectorizer()

            matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            df = pd.DataFrame(matrix.toarray(), index=[f'Doc_{i+1}' for i in range(matrix.shape[0])], columns=feature_names)
            st.subheader(f'{matrix_type} 矩阵（{matrix.shape[0]} 行 x {matrix.shape[1]} 列）')
            st.dataframe(df.style.format('{:.4f}'))

            if matrix_type == 'TF-IDF':
                sums = np.asarray(matrix.sum(axis=0)).ravel()
                top_n = 5
                top_idx = np.argsort(sums)[::-1][:top_n]
                top_keywords = [(feature_names[idx], sums[idx]) for idx in top_idx]

                st.subheader('TF-IDF 权重最高的 5 个关键词')
                st.table(pd.DataFrame(top_keywords, columns=['关键词', '总权重']))
            else:
                st.info('当前为 One-hot 表示，关键词可视化未包含 TF-IDF 权重选择。')

            if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
                svd = TruncatedSVD(n_components=2, random_state=42)
                term_vectors = svd.fit_transform(matrix.T)
                term_df = pd.DataFrame(term_vectors, columns=['x', 'y'], index=feature_names)

                st.subheader('LSA（TruncatedSVD）词向量 2D 可视化')
                import altair as alt

                scatter = (
                    alt.Chart(term_df.reset_index())
                    .mark_circle(size=100)
                    .encode(
                        x='x',
                        y='y',
                        tooltip=['index', 'x', 'y']
                    )
                    .properties(width=800, height=500)
                )
                text = (
                    alt.Chart(term_df.reset_index())
                    .mark_text(dx=7, dy=-7, fontSize=12)
                    .encode(x='x', y='y', text='index')
                )

                st.altair_chart(scatter + text, use_container_width=True)
            else:
                st.warning('句子数或词汇维度小于 2，无法进行 LSA 可视化。')

with tabs[1]:
    st.header('模块 2：语义词向量（Word2Vec 实时训练与测试）')

    text_for_w2v = st.text_area('输入英文语料（可多句，用于训练 Word2Vec）', height=220, placeholder='输入英文文本，例如：\nThis is the first sentence.\nHere is another sentence.')

    w2v_arch = st.radio('选择 Word2Vec 训练架构', ['CBOW (sg=0)', 'Skip-Gram (sg=1)'])
    w2v_window = st.slider('窗口 window 大小', 2, 10, 5)
    w2v_size = st.slider('词向量维度', 50, 300, 100)
    w2v_epochs = st.slider('训练轮数（epochs）', 5, 50, 20)

    if text_for_w2v:
        sentences_w2v = [s.strip() for s in nltk.sent_tokenize(text_for_w2v) if s.strip()]
        tokenized = [nltk.word_tokenize(s.lower()) for s in sentences_w2v]

        if len(tokenized) == 0:
            st.warning('请提供至少一句有效英文语料来训练 Word2Vec。')
        else:
            from gensim.models import Word2Vec

            sg = 0 if w2v_arch.startswith('CBOW') else 1
            model = Word2Vec(sentences=tokenized, vector_size=w2v_size, window=w2v_window, sg=sg, min_count=1, epochs=w2v_epochs)
            st.session_state['w2v_model'] = model

            st.success('Word2Vec 模型训练完成。')
            st.write('架构：', 'CBOW' if sg == 0 else 'Skip-Gram', '，window：', w2v_window, '，向量维度：', w2v_size, '，epochs：', w2v_epochs)

            query_word = st.text_input('输入要查找相似词的单词', value='boat')

            if query_word:
                query_word = query_word.lower()
                if query_word in model.wv:
                    similar_words = model.wv.most_similar(query_word, topn=5)
                    st.subheader(f'与 "{query_word}" 余弦相似度最高的 5 个词')
                    st.table(pd.DataFrame(similar_words, columns=['词', '相似度']))
                else:
                    st.error(f'词 "{query_word}" 不在词汇表中，请更换词或补充语料。')

with tabs[2]:
    st.header('模块 3：预训练 GloVe 模型与词类比')

    st.markdown('加载预训练 glove-twitter-25（轻量级，适合演示）。')
    import gensim.downloader as api
    @st.cache_resource
    def load_glove_model():
        return api.load('glove-twitter-25')

    glove_model = load_glove_model()

    st.write('词表大小：', len(glove_model.key_to_index))

    col1, col2, col3 = st.columns(3)
    with col1:
        word_a = st.text_input('A (例如：king)')
    with col2:
        word_b = st.text_input('B (例如：man)')
    with col3:
        word_c = st.text_input('C (例如：woman)')

    if word_a and word_b and word_c:
        word_a = word_a.strip().lower()
        word_b = word_b.strip().lower()
        word_c = word_c.strip().lower()

        if all(w in glove_model.key_to_index for w in [word_a, word_b, word_c]):
            result = glove_model.most_similar(positive=[word_a, word_c], negative=[word_b], topn=5)
            st.subheader('类比结果 A-B+C:')
            st.table(pd.DataFrame(result, columns=['词', '相似度']))
        else:
            missing=[w for w in [word_a, word_b, word_c] if w not in glove_model.key_to_index]
            st.error('以下词不在词表中: ' + ', '.join(missing))

    st.markdown('---')
    st.subheader('词义相似度计算')

    sim_w1 = st.text_input('相似度词1', value='king', key='sim_w1')
    sim_w2 = st.text_input('相似度词2', value='queen', key='sim_w2')

    if sim_w1 and sim_w2:
        sim_w1 = sim_w1.strip().lower()
        sim_w2 = sim_w2.strip().lower()
        if sim_w1 in glove_model.key_to_index and sim_w2 in glove_model.key_to_index:
            score = glove_model.similarity(sim_w1, sim_w2)
            st.write(f'"{sim_w1}" 与 "{sim_w2}" 的词义相似度：{score:.4f}')
        else:
            missing=[w for w in [sim_w1, sim_w2] if w not in glove_model.key_to_index]
            st.error('以下词不在词表中: ' + ', '.join(missing))

with tabs[3]:
    st.header('模块 4：FastText 与句子级表示（Sent2Vec）')

    text_for_ft = st.text_area('输入英文语料（可多句，用于训练 FastText/Word2Vec）', height=220, placeholder='输入英文文本，例如：\nThis is the first sentence.\nAnother one here.')

    if text_for_ft:
        sentences_ft = [s.strip() for s in nltk.sent_tokenize(text_for_ft) if s.strip()]
        tokenized_ft = [nltk.word_tokenize(s.lower()) for s in sentences_ft]

        from gensim.models import FastText

        ft_model = FastText(sentences=tokenized_ft, vector_size=100, window=5, min_count=1, epochs=20)
        st.session_state['ft_model'] = ft_model

        st.success('FastText 模型训练完成。')

        oov_word = st.text_input('OOV 测试词（如 computeer）', value='computeer')
        if oov_word:
            oov_word = oov_word.strip().lower()

            # Word2Vec 旧模型存在则尝试提取
            if 'w2v_model' in st.session_state:
                try:
                    w2v_vec = st.session_state['w2v_model'].wv[oov_word]
                    st.write('Word2Vec 词向量已获取（通常情况下 OOV 不支持）。')
                except KeyError:
                    st.warning('Word2Vec: 未登录词')
                    w2v_vec = None
            else:
                st.info('Word2Vec 模型未在会话中找到，请先在第二个标签页训练 Word2Vec。')
                w2v_vec = None

            # FastText 总是能返回向量
            try:
                ft_vec = ft_model.wv[oov_word]
                st.write('FastText 已成功计算 OOV 词向量。')
                similar_fasttext = ft_model.wv.most_similar(oov_word, topn=5)
                st.subheader(f'FastText 对 "{oov_word}" 的相似词')
                st.table(pd.DataFrame(similar_fasttext, columns=['词', '相似度']))
            except Exception as ex:
                st.error('FastText 处理 OOV 词出错：' + str(ex))

        st.markdown('---')

        st.subheader('Sent2Vec (Average Pooling) 语义相似度')
        sent1 = st.text_area('句子1', value='This is a sentence about machine learning.', key='sent2vec_1')
        sent2 = st.text_area('句子2', value='This text describes deep learning algorithms.', key='sent2vec_2')

        if sent1 and sent2:
            tokens1 = [w.lower() for w in nltk.word_tokenize(sent1) if w.isalpha()]
            tokens2 = [w.lower() for w in nltk.word_tokenize(sent2) if w.isalpha()]

            vec1 = np.mean([ft_model.wv[t] for t in tokens1], axis=0) if tokens1 else np.zeros(ft_model.vector_size)
            vec2 = np.mean([ft_model.wv[t] for t in tokens2], axis=0) if tokens2 else np.zeros(ft_model.vector_size)

            from numpy.linalg import norm
            if norm(vec1) > 0 and norm(vec2) > 0:
                cosine_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
                st.write(f'句子余弦相似度：{cosine_sim:.4f}')
            else:
                st.warning('其中一个句子没有有效词向量，无法计算相似度。')
    else:
        st.info('请在上方输入语料以训练 FastText 模型。')

