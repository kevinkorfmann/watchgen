import os
import sys

project = "The Watchmaker's Guide to Population Genetics"
copyright = '2026, Kevin Korfmann'
author = "The Watchmaker's Guide Contributors"
release = '0.1.0'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

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
latex_additional_files = ["_static/logo.png", "_static/cover.pdf"]
latex_engine = "xelatex"
latex_elements = {
    "preamble": r"""
\hypersetup{pdfauthor={Kevin Korfmann}}
\usepackage{tikz}
\usepackage{pdfpages}
\setcounter{tocdepth}{3}
""",
    "extraclassoptions": "openright,twoside",
    "maketitle": r"""
\makeatletter
%% Full-bleed cover page via pdfpages (guaranteed no margins)
\includepdf[pages=1,fitpaper=false]{cover.pdf}
%% Standard Sphinx title page
\let\sphinxrestorepageanchorsetting\relax
\ifHy@pageanchor\def\sphinxrestorepageanchorsetting{\Hy@pageanchortrue}\fi
\hypersetup{pageanchor=false}%
\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \noindent\rule{\textwidth}{1pt}\par
    \begingroup
     \def\endgraf{ }\def\and{\& }%
     \pdfstringdefDisableCommands{\def\\{, }}%
     \hypersetup{pdfauthor={\@author}, pdftitle={\@title}}%
    \endgroup
  \begin{flushright}%
    \sphinxlogo
    \py@HeaderFamily
    {\Huge \@title \par}
    {\itshape\LARGE \py@release\releaseinfo \par}
    \vfill
    {\LARGE
      \begin{tabular}[t]{c}
        \@author
      \end{tabular}\kern-\tabcolsep
      \par}
    \vfill\vfill
    {\large
     \@date \par
    }%
  \end{flushright}%
  \vspace*{\fill}
  \noindent\includegraphics[width=3cm]{logo.png}
\end{titlepage}%
\setcounter{footnote}{0}%
\let\thanks\relax\let\maketitle\relax
\clearpage
%% Epigraph page (Red Queen)
\thispagestyle{empty}
\vspace*{\fill}
\begin{center}
\begin{minipage}{0.75\textwidth}
\Large\itshape
``Die Tat ist alles, nichts der Ruhm.''
\end{minipage}

\vspace{1.5em}
\normalsize\upshape
--- Johann Wolfgang von Goethe, \textit{Faust II}
\end{center}
\vspace*{\fill}
\clearpage
%% Preface (unnumbered, front matter -- before Chapter 1)
\if@openright\cleardoublepage\else\clearpage\fi
\chapter*{Preface}
\addcontentsline{toc}{chapter}{Preface}

This book was written with the assistance of Anthropic's Claude Opus 4.6, largely
within the one-million-token context window of Claude Code. That disclosure made,
let me explain what this project actually is and why it exists. It is also,
frankly, an experiment in agentic coding --- an attempt to see how far a single
researcher can push the boundaries of technical writing and software development
when working in close collaboration with an AI agent. The chapters, derivations,
implementations, and tests were produced through an iterative dialogue between
human intent and machine capability, and the result is as much a proof of concept
for that workflow as it is a textbook.

Population genetics is blessed with powerful algorithms --- but cursed with
inaccessible ones. Many of the field's most important methods live inside papers
and codebases that assume years of specialized training to read, let alone
reimplement. They rarely come with manuals, guided tours, or on-ramps for the
curious outsider --- or even for the insider who works on a different corner of
the field. This book is an attempt to change that: to make the algorithms not
only open but genuinely \emph{accessible}, with all the prerequisites laid out and
every derivation shown in full.

I assembled these chapters during my transition from the University of Oregon to
the University of Pennsylvania, at my own expense and in my own time. The book is
freely available because I believe science should be. It was not, however, free to
create --- and I mention this only to underscore that the motivation was personal
before it was practical. I wanted to understand these algorithms more deeply myself.
Writing them out, gear by gear, was the surest way I knew how.

Nothing here is meant to diminish the original work. Each algorithm in this book
represents a serious intellectual achievement --- the kind that earns PhD titles and
advances entire subfields. The goal is translation, not judgment: to take ideas that
were expressed for expert audiences and re-express them for anyone willing to learn.
The content may contain errors --- mathematical, conceptual, or otherwise --- and
should ideally be read in combination with the original journal articles, which
remain the authoritative source for each method.

The book is accompanied by \texttt{watchgen}, a Python package that provides minimal,
self-contained implementations of every algorithm covered. These mini implementations
are not production tools; they are pedagogical companions --- small enough to read in
one sitting, complete enough to run on toy examples, and tested enough to give you
confidence that the math on the page actually works. Think of them as the movements
you build on the workbench: not meant for sale, but meant to teach your hands
what your eyes have read.

Finally, this project is an open invitation. I welcome collaborators who want to
cross-check derivations, correct mistakes, improve explanations, add chapters, or
simply point out where things could be clearer. The ambition is a living resource
that grows more accurate and more useful over time, built by the community it is
meant to serve.

\medskip
\noindent\emph{Looking ahead.}
The mini implementations in \texttt{watchgen} are pedagogical ---
deliberately simple, deliberately slow. But the landscape is shifting fast. AI
models are growing more capable at an extraordinary pace, and within the next year
it may become realistic to go further: to use these same models to produce a
unified, production-grade software package that brings the algorithms covered here
under a single roof --- correct, tested, interoperable, and maintained. This book,
with its explicit derivations and reference implementations, is designed to serve
as the foundation for exactly that kind of effort.

\medskip
\noindent\emph{On versioning.}
This is version 0.1 --- an unverified draft. No chapter has yet
been reviewed by a domain expert, and I make no claim that any derivation is free
of error. Future versions will name the individuals who have verified each chapter,
and contributors who substantially improve the content --- whether by correcting
proofs, rewriting sections, or adding new chapters --- will be invited as co-authors.
Science is a collective enterprise; this book should be too.

\vfill
\noindent Kevin Korfmann\\
Philadelphia, 2026

\vspace{1em}\nopagebreak
\noindent\small\textcopyright\ 2026 Kevin Korfmann.
Prose and documentation licensed under
\href{https://creativecommons.org/licenses/by-nc-sa/4.0/}{CC BY-NC-SA 4.0}
(non-commercial, share-alike).
Source code licensed under the MIT License.
\normalsize

\clearpage
\if@openright\cleardoublepage\else\clearpage\fi
\sphinxrestorepageanchorsetting
\makeatother
""",
}
