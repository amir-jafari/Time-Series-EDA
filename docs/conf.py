# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))

# Ensure matplotlib uses a non-interactive backend for notebook execution
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------------------------------------------------------
# Copy external artefacts into the docs source tree at build time.
# Notebooks and HTML reports stay in their natural locations; Sphinx gets
# build-time copies (tracked by .gitignore).
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_DOCS = Path(__file__).parent

# Notebook → docs/examples/
_NB_SRC = _ROOT / "notebooks" / "Global_Air_Pollution_EDA.ipynb"
_NB_DST = _DOCS / "examples"
if _NB_SRC.exists():
    _NB_DST.mkdir(exist_ok=True)
    shutil.copy2(_NB_SRC, _NB_DST / _NB_SRC.name)

# HTML reports → docs/_static/reports/
_HTML_SRC = _ROOT / "notebooks" / "html"
_HTML_DST = _DOCS / "_static" / "reports"
if _HTML_SRC.exists():
    _HTML_DST.mkdir(parents=True, exist_ok=True)
    for _f in _HTML_SRC.glob("*.html"):
        shutil.copy2(_f, _HTML_DST / _f.name)

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = "tseda"
author    = "Amirhossein Jafari"
copyright = "2026, Amirhossein Jafari"
release   = "0.1.2"
version   = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings automatically
    "sphinx.ext.napoleon",      # NumPy & Google docstring styles
    "sphinx.ext.viewcode",      # [source] links on every page
    "sphinx.ext.intersphinx",   # cross-link numpy, pandas, scipy, matplotlib
    "sphinx.ext.mathjax",       # LaTeX math rendering
    "sphinx.ext.githubpages",   # .nojekyll for GitHub Pages
    "sphinx_copybutton",        # copy button on code blocks
    "sphinx_design",            # grid / card directives
    "myst_parser",              # include .md files as pages
    "nbsphinx",                 # render Jupyter notebooks
]

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__init__, __repr__, __len__, __contains__, __getitem__",
    "exclude-members":  "maketrans",
}
autodoc_typehints        = "description"
autodoc_typehints_format = "short"

# ---------------------------------------------------------------------------
# Napoleon
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = True

# ---------------------------------------------------------------------------
# nbsphinx — use stored outputs when present, execute if missing
# ---------------------------------------------------------------------------
nbsphinx_execute         = "auto"
nbsphinx_timeout         = 300
nbsphinx_kernel_name     = "python3"
nbsphinx_prolog          = ""

# ---------------------------------------------------------------------------
# MyST parser
# ---------------------------------------------------------------------------
myst_enable_extensions = ["dollarmath"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
# nbsphinx registers .ipynb automatically — do NOT add it to source_suffix

# ---------------------------------------------------------------------------
# Copy-button — skip prompts
# ---------------------------------------------------------------------------
copybutton_prompt_text      = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3/",        None),
    "numpy":      ("https://numpy.org/doc/stable/",     None),
    "pandas":     ("https://pandas.pydata.org/docs/",   None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/",    None),
}

# ---------------------------------------------------------------------------
# MathJax
# ---------------------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "inlineMath":  [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# ---------------------------------------------------------------------------
# HTML output — sphinx_rtd_theme matching TimeSeries Toolbox style
# ---------------------------------------------------------------------------
html_theme       = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files   = ["css/custom.css"]
html_title       = "tseda — Time Series EDA"
html_short_title = "tseda"

html_theme_options = {
    "logo_only":                  False,
    "prev_next_buttons_location": "bottom",
    "style_external_links":       True,
    "collapse_navigation":        False,
    "sticky_navigation":          True,
    "navigation_depth":           4,
    "includehidden":              True,
    "titles_only":                False,
}

# Show "Edit on GitHub" links
html_context = {
    "display_github": True,
    "github_user":    "amir-jafari",
    "github_repo":    "Time-Series-EDA",
    "github_version": "main",
    "conf_py_path":   "/docs/",
}

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
templates_path   = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "examples/README.md",
]