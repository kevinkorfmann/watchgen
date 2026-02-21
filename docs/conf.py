import os
import sys

project = "The Watchmaker's Guide to Population Genetics"
copyright = '2024, The Watchmaker\'s Guide Contributors'
author = "The Watchmaker's Guide Contributors"
release = '0.1.0'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/logo.png'

html_theme_options = {
    'logo_only': False,
    'version_selector': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_title = "The Watchmaker's Guide to Population Genetics"
html_short_title = "Watchmaker's Guide"

mathjax3_config = {
    'tex': {
        'macros': {
            'RR': r'\mathbb{R}',
            'NN': r'\mathbb{N}',
            'EE': r'\mathbb{E}',
            'PP': r'\mathbb{P}',
        }
    }
}

pygments_style = 'friendly'
html_show_sourcelink = False

# PDF/LaTeX: book format, author for title page and PDF metadata
latex_author = "Kevin Korfmann"
latex_docclass = {"manual": "book"}
latex_documents = [
    ("index", "watchmakers-guide.tex", project, latex_author, "manual"),
]
latex_elements = {
    "preamble": r"""
\hypersetup{pdfauthor={Kevin Korfmann}}
""",
    "extraclassoptions": "openright,twoside",
}
