# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information

project = "hyperwave"
copyright = "2024, SPINS Photonics Inc"
author = "SPINS Photonics Inc"

release = ""
version = ""

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinx_remove_toctrees",
    "sphinx_copybutton",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

source_suffix = [".rst", ".ipynb", ".md"]

# The main toctree document.
main_doc = "index"

# -- Options for HTML output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/spinsphotonics/hyperwave",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
}

# Tell sphinx autodoc how to render type aliases.
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
}
