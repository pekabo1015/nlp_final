from __future__ import annotations

import os
import runpy
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import streamlit as st


def _purge_modules(prefixes: list[str]) -> None:
    to_delete = []
    for name in sys.modules.keys():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                to_delete.append(name)
                break
    for name in to_delete:
        sys.modules.pop(name, None)


@contextmanager
def _patched_streamlit_config():
    original_set_page_config = st.set_page_config

    def _safe_set_page_config(*args, **kwargs):
        return None

    st.set_page_config = _safe_set_page_config
    try:
        yield
    finally:
        st.set_page_config = original_set_page_config


@contextmanager
def _temporary_script_context(script_path: Path):
    previous_cwd = Path.cwd()
    script_dir = script_path.parent
    inserted_path = False

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
        inserted_path = True

    os.chdir(script_dir)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if inserted_path:
            sys.path.remove(str(script_dir))


def run_legacy_streamlit_script(script_path: str | Path) -> None:
    path = Path(script_path).resolve()
    if not path.exists():
        st.error(f"未找到脚本：{path}")
        return

    with _patched_streamlit_config():
        with _temporary_script_context(path):
            _purge_modules(["nltk", "benepar"])
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit:
                return
            except Exception as exc:
                st.error(f"板块加载失败：{exc}")
                with st.expander("查看详细报错", expanded=False):
                    st.code(traceback.format_exc(), language="text")
