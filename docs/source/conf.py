# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
import six
import sphinx_rtd_theme
from sphinx.ext.autodoc import between
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'EdgeL3'
copyright = '2019, Sangeeta Kumari'
author = 'Sangeeta Kumari'


# -- Mock dependencies
if six.PY3:
    from unittest.mock import MagicMock
else:
    from mock import Mock as MagicMock

class Mock(MagicMock):
    @classmethod
    def getattr(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'numpy', 'soundfile', 'resampy', 'keras', 'tensorflow',
    'kapre', 'kapre.time_frequency', 'keras.layers', 'keras.models',
    'keras.regularizers'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# The full version, including alpha/beta/rc tags
import imp
openl3_version = imp.load_source('edgel3.version', '../../edgel3/version.py')
version = edgel3_version.short_version
release = edgel3_version.version

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_issues',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
