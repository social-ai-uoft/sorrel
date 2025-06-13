"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sorrel"
copyright = "2025, Yibing Ju, Rebekah Gelpi, Ethan Jackson, Shon Verch, Yikai Tang, Claas Voelcker, Wil Cunningham"
author = "Yibing Ju, Rebekah Gelpi, Ethan Jackson, Shon Verch, Yikai Tang, Claas Voelcker, Wil Cunningham"
release = "1.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "autodoc2",
    # "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
]

# extensions for math support and other qol in markdown files
# see https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autodoc2_packages = [
    {
        "path": "../../sorrel",
        "auto_mode": False,
    },
]
autodoc2_module_summary = False


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
