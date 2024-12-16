import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'Dopterian+'
copyright = '2024, Diego Maldonado, Pierluigi Cerulo, Ana Paulino-Afonso'
author = 'Diego Maldonado, Pierluigi Cerulo, Ana Paulino-Afonso'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_context = {
    "display_github": True,  
    "github_user": "dimaldonado",  
    "github_repo": "dopterian-plus",  
    "github_version": "main",  
    "doc_path": "docs/source/",  
}