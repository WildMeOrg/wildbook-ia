# -*- coding: utf-8 -*-
from sphinx.ext.autodoc import between
import alabaster  # NOQA
# import sys
# sys.path.insert(0, os.path.abspath('.'))

try:
    import wbia
except ImportError:
    raise RuntimeError("Wildbook-IA (the 'wbia' package) must be installed")

autosummary_generate = True

modindex_common_prefix = ['_']

# -- Project information -----------------------------------------------------

project = 'Wildbook Image Analysis (IA)'
copyright = '2020, Wild Me'
author = 'Jon Crall, Jason Parham, WildMe Developers'

version = release = wbia.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'alabaster',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    # For LaTeX
    'sphinx.ext.imgmath',
    # For Google Sytle Docstrs
    # https://pypi.python.org/pypi/sphinxcontrib-napoleon
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

# -- Theme options -----------------------------------------------------------

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect('autodoc-process-docstring', between('^.*IGNORE.*$', exclude=True))
    return app
