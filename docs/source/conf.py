from __future__ import annotations

from pathlib import Path


project = "MITRE Capstone GNN"
author = "MITRE Capstone"
release = "1.0"

extensions = [
    "autoapi.extension",
]

source_suffix = ".rst"
master_doc = "index"
language = "en"

html_theme = "alabaster"
html_static_path: list[str] = []
suppress_warnings = [
    "ref.python",
]

repo_root = Path(__file__).resolve().parents[2]
autoapi_type = "python"
autoapi_dirs = [
    str(repo_root / ".sphinx-api"),
]
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_ignore = [
    "*/__pycache__/*",
]
