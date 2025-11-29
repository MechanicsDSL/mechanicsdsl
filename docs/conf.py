import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('../src'))

import mechanics_dsl  # This verifies we can import your package

# -- Project information -----------------------------------------------------
project = 'MechanicsDSL'
copyright = f'{date.today().year}, Noah Parsons'
author = 'Noah Parsons'
version = mechanics_dsl.__version__
release = mechanics_dsl.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from your code docstrings
    'sphinx.ext.napoleon',     # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.mathjax',      # Render LaTeX math
    'myst_parser',             # Support Markdown (.md) files
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # Add '_static' if you have custom CSS

# -- Extension configuration -------------------------------------------------
# Allow both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc settings
autodoc_member_order = 'bysource'
