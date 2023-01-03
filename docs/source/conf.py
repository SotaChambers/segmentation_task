# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# ---------- 追加 ---------- #
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
# ---------- /追加 ---------- #

project = 'sphinx_test'
copyright = '2023, sn'
author = 'sn'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# ---------- 追加 ---------- #

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon'
]
# ---------- /追加 ---------- #

templates_path = ['_templates']
exclude_patterns = []

language = 'ja'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# ---------- 追加 ---------- #
html_theme = 'sphinx_rtd_theme'
# ---------- /追加 ---------- #
html_static_path = ['_static']
