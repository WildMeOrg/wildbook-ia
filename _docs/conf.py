# -*- coding: utf-8 -*-
from sphinx.ext.autodoc import between
import alabaster  # NOQA
import sys
import os

sys.path.append(sys.path.insert(0, os.path.abspath('../')))

autosummary_generate = True

modindex_common_prefix = ['_']

master_doc = 'index'

html_theme = 'alabaster'

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# -- Project information -----------------------------------------------------

project = 'wbia'
copyright = '2020, Wild Me'
author = 'Jon Crall, Jason Parham, WildMe Developers'

# The short X.Y version
version = '3.0.1'

# The full version, including alpha/beta/rc tags
release = '3.0.1'


# -- General configuration ---------------------------------------------------

MOCK_MODULES = []
if len(MOCK_MODULES) > 0:
    import mock

    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = mock.Mock()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    # For LaTeX
    'sphinx.ext.imgmath',
    # For Google Sytle Docstrs
    # https://pypi.python.org/pypi/sphinxcontrib-napoleon
    'sphinx.ext.napoleon',
    'alabaster',
]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect('autodoc-process-docstring', between('^.*IGNORE.*$', exclude=True))
    return app
