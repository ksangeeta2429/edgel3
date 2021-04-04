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
import sklearn.decomposition
import sphinx_rtd_theme
from sphinx.ext.autodoc import between
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../'))
#sys.path.insert(0, os.path.abspath('../../edgel3/edgel3'))

# -- Project information -----------------------------------------------------

project = 'edgel3'
copyright = '2019, Sangeeta Srivastava'
author = 'Sangeeta Srivastava'

if sys.version_info[:2] >= (3, 3):
    import platform
    from importlib.machinery import SourceFileLoader
    def load_source(name, path):
        if not os.path.exists(path):
            return {}
        return vars(SourceFileLoader(name, path).load_module())
else:
    import imp
    def load_source(name, path):
        if not os.path.exists(path):
            return {}
        return vars(imp.load_source(name, path))

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
edgel3_version = load_source('edgel3.version', '../../edgel3/version.py')
version = edgel3_version['short_version']
release = edgel3_version['version']

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
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx_issues',
]

# Napoleon settings
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

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
