# -*- coding: utf-8 -*-
from sphinx.ext.autodoc import between
import alabaster
import sys
import os

# Dont parse IBEIS args
os.environ['IBIES_PARSE_ARGS'] = 'OFF'
os.environ['UTOOL_AUTOGEN_SPHINX_RUNNING'] = 'ON'

sys.path.append(sys.path.insert(0, os.path.abspath('../')))

autosummary_generate = True

modindex_common_prefix = ['_']
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os  # NOQA
# import sys  # NOQA
# sys.path.insert(0, '/Users/jason.parham/code/ibeis/ibeis')

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
author = 'Jason Parham, Jon Crall, Drew Blount, Michael Mulich, Karen Chan'

# The short X.Y version
version = '2.2.4'
# The full version, including alpha/beta/rc tags
release = '2.2.4'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
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
