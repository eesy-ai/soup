# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os 
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SOUP'
copyright = '2024, Eesy-Innovation GmbH'
author = 'Eesy-Innovation GmbH'
release = '1.0.0'

# -- General configuration -------------------------------
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.todo"]

napoleon_custom_sections = [('Returns', 'params_style')]
autodoc_default_options = {"members": True, "undoc-members": True, "private-members": True}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -----------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for Latex output ----------------------------
latex_documents = [('index', 'Documentation.tex', project, author, 'manual')]
latex_elements = {
    'preamble': r"""
        \usepackage{eesyDocStyle}
    """,

    'maketitle': r"""
        \makeEesyTitlePage{%s}{%s}
    """ % (project, release),
}

latex_additional_files = [f"_static/{file}" for file in os.listdir("_static")]