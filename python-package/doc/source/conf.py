"""Sphinx configuration for the pycasso documentation."""

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

# Autodoc needs signatures and docstrings but does not construct Solver.
# This makes a native library optional for a clean documentation build.
os.environ.setdefault("PICASSO_BUILD_DOC", "1")

project = "pycasso"
author = "PICASSO authors"
copyright = "2017-2026, PICASSO authors"

release = (PACKAGE_ROOT / "pycasso" / "VERSION").read_text(
    encoding="utf-8").strip()
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
root_doc = "index"
master_doc = "index"
source_suffix = ".rst"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = []
html_static_path = []
autodoc_member_order = "bysource"
autodoc_typehints = "none"

try:
    import sphinx_rtd_theme  # noqa: F401
except ImportError:
    html_theme = "alabaster"
else:
    html_theme = "sphinx_rtd_theme"

html_title = f"pycasso {release}"
htmlhelp_basename = "pycassodoc"

latex_documents = [
    ("index", "pycasso.tex", "pycasso Documentation",
     "PICASSO authors", "manual"),
]
man_pages = [
    ("index", "pycasso", "PICASSO sparse regularization-path solver",
     [author], 1),
]
