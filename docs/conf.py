# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = "tseda"
copyright = "2026, Amirhossein Jafari"
author    = "Amirhossein Jafari"
release   = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",           # auto-generate docs from docstrings
    "sphinx.ext.viewcode",          # add [source] links
    "sphinx.ext.intersphinx",       # cross-ref to numpy / pandas / scipy docs
    "sphinx.ext.napoleon",          # NumPy & Google docstring support
    "sphinx_copybutton",            # copy button on code blocks
    "sphinx.ext.mathjax",           # LaTeX math rendering
    "sphinx.ext.githubpages",       # .nojekyll file for GitHub Pages
]

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__init__, __repr__, __len__, __contains__, __getitem__",
    "exclude-members":  "maketrans",
}
autodoc_typehints        = "description"   # types in Parameters / Returns sections
autodoc_typehints_format = "short"

# ---------------------------------------------------------------------------
# Intersphinx — links to upstream package docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3/", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output — sphinx_rtd_theme (mirrors https://amir-jafari.github.io/TimeSeries/)
# ---------------------------------------------------------------------------
html_theme       = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files   = ["css/custom.css"]

html_theme_options = {
    # Navigation
    "collapse_navigation":   False,
    "sticky_navigation":     True,
    "navigation_depth":      4,
    "includehidden":         True,
    "titles_only":           False,

    # Branding
    "logo_only":             False,
    "prev_next_buttons_location": "bottom",
    "style_external_links":  True,

    # GitHub integration
    "style_nav_header_background": "#2c3e50",
}

html_title      = "tseda — Time Series EDA"
html_short_title = "tseda"

# Sidebar logo / favicon (add files to docs/_static/ when available)
# html_logo   = "_static/tseda_logo.png"
# html_favicon = "_static/favicon.ico"

# Show "Edit on GitHub" links
html_context = {
    "display_github":  True,
    "github_user":     "amir-jafari",
    "github_repo":     "Time-Series-EDA",
    "github_version":  "main",
    "conf_py_path":    "/docs/",
}