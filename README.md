# 自然语言处理交互式 Web 分析系统

基于 `Python + Streamlit` 开发的课程综合展示系统，将 `A1` 到 `A9` 九个自然语言处理实验整合到同一个网页中，并支持通过 `GitHub + Streamlit Cloud` 在线部署。

## 项目结构

```text
hw/
├─ app.py
├─ requirements.txt
├─ shared/
├─ A1/
├─ A2/
├─ A3/
├─ A4/
├─ A5/
├─ A6/
├─ A7/
├─ A8/
└─ A9/
```

## 本地运行

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动系统：

```bash
streamlit run app.py
```

## Streamlit Cloud 部署

1. 将整个项目上传到同一个 GitHub 仓库。
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)。
3. 选择 `New app`。
4. 选择你的 GitHub 仓库和分支。
5. 将主文件设置为：

```text
app.py
```

6. 点击部署，等待依赖安装完成。

## 部署注意事项

- 根目录 `requirements.txt` 是云端安装依赖的唯一入口，请不要删除。
- 某些板块涉及 `spaCy`、`transformers`、`torch` 等模型，首次运行时可能较慢。
- 某些大模型或在线下载资源在云端网络环境下可能加载失败，系统会在对应板块显示提示信息。
- 如果后续你有自己的 GitHub 仓库地址，可以把它补充到系统首页或本 README 中。

## 当前整合方式

- 统一入口文件为 `app.py`。
- 左侧边栏用于切换首页与 9 个实验板块。
- 每个板块优先复用原始实验脚本。
- 通过 `shared/legacy_runner.py` 兼容原有独立 `Streamlit` 脚本，减少对作业源码的侵入式修改。
