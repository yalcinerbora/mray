# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# TODO: Fetch this from CMake like "release"
#project = 'MRay'
copyright = '%Y, Bora Yalciner'
author = 'Bora Yalciner'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = ['myst_parser',
              'sphinxcontrib.mermaid',
              'sphinx_copybutton']
templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "MRay Documentation"
html_css_files = ['custom.css']
html_js_files  = [
    "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js",
    'cBox.js',
    'mray_mermaid_init.js'
]

html_favicon = '_static/mray_cbox.png'
html_baseurl = '/Docs/'

html_theme_options = {
    "repository_url": "https://github.com/yalcinerbora/mray",
    "use_repository_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 3,

    "show_navbar_depth": 2,
    "max_navbar_depth": 3,

    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",


    'collapse_navigation': True,
    'navigation_depth': 4
}

html_sidebars = {
    "**": [
           "path_trace.html",
           "navbar-logo.html",
           "search-button-field.html",
           "sbt-sidebar-nav.html"]
}

# -- Options for Myst markdown parser -----------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "attrs_inline",
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# -- Options for pygments (syntax coloring) -----------------------------------------


# -- Options for Mermaid  ----------------------

# -- Options for LateX output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output
latex_engine = 'pdflatex'
