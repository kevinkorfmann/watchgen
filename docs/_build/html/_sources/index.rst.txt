.. _index:

====================================================
The Watchmaker's Guide to Population Genetics
====================================================

   *A watchmaker doesn't just use a watch -- they understand every gear, every spring, every mechanism. Nothing is a black box.*

.. only:: html

   .. image:: https://img.shields.io/badge/philosophy-build%20it%20yourself-blue
      :alt: Build It Yourself

   .. grid:: 2
      :gutter: 2

      .. grid-item::

         .. button-link:: https://github.com/kevinkorfmann/watchgen/raw/main/docs/_build/latex/watchmakers-guide.pdf
            :color: primary
            :expand:

            Download PDF

      .. grid-item::

         .. button-link:: https://www.paypal.com/donate/?hosted_button_id=VTASTXN2KAFJQ
            :color: success
            :expand:

            Support with PayPal

   .. note::

      This book has not yet been assigned a version number. A citable release is planned for the coming days.
      In the meantime, please cite by URL and access date.

   .. raw:: html

      <hr>

   .. rubric:: Preface

   This book was written with the assistance of Anthropic's Claude Opus 4.6, largely
   within the one-million-token context window of Claude Code. That disclosure made,
   let me explain what this project actually is and why it exists.

   Population genetics is blessed with powerful algorithms -- but cursed with
   inaccessible ones. Many of the field's most important methods live inside papers
   and codebases that assume years of specialized training to read, let alone
   reimplement. They rarely come with manuals, guided tours, or on-ramps for the
   curious outsider -- or even for the insider who works on a different corner of
   the field. This book is an attempt to change that: to make the algorithms not
   only open but genuinely *accessible*, with all the prerequisites laid out and
   every derivation shown in full.

   I assembled these chapters during my transition from the University of Oregon to
   the University of Pennsylvania, at my own expense and in my own time. The book is
   freely available because I believe science should be. It was not, however, free to
   create -- and I mention this only to underscore that the motivation was personal
   before it was practical. I wanted to understand these algorithms more deeply myself.
   Writing them out, gear by gear, was the surest way I knew how.

   Nothing here is meant to diminish the original work. Each algorithm in this book
   represents a serious intellectual achievement -- the kind that earns PhD titles and
   advances entire subfields. The goal is translation, not judgment: to take ideas that
   were expressed for expert audiences and re-express them for anyone willing to learn.
   The content may contain errors -- mathematical, conceptual, or otherwise -- and
   should ideally be read in combination with the original journal articles, which
   remain the authoritative source for each method.

   The book is accompanied by ``watchgen``, a Python package that provides minimal,
   self-contained implementations of every algorithm covered. These mini implementations
   are not production tools; they are pedagogical companions -- small enough to read in
   one sitting, complete enough to run on toy examples, and tested enough to give you
   confidence that the math on the page actually works. Think of them as the movements
   you build on the workbench: not meant for sale, but meant to teach your hands
   what your eyes have read.

   Finally, this project is an open invitation. I welcome collaborators who want to
   cross-check derivations, correct mistakes, improve explanations, add chapters, or
   simply point out where things could be clearer. The ambition is a living resource
   that grows more accurate and more useful over time, built by the community it is
   meant to serve.

   *Looking ahead.* The mini implementations in ``watchgen`` are pedagogical --
   deliberately simple, deliberately slow. But the landscape is shifting fast. AI
   models are growing more capable at an extraordinary pace, and within the next year
   it may become realistic to go further: to use these same models to produce a
   unified, production-grade software package that brings the algorithms covered here
   under a single roof -- correct, tested, interoperable, and maintained. This book,
   with its explicit derivations and reference implementations, is designed to serve
   as the foundation for exactly that kind of effort.

   *On versioning.* This is version 0.1 -- an unverified draft. No chapter has yet
   been reviewed by a domain expert, and I make no claim that any derivation is free
   of error. Future versions will name the individuals who have verified each chapter,
   and contributors who substantially improve the content -- whether by correcting
   proofs, rewriting sections, or adding new chapters -- will be invited as co-authors.
   Science is a collective enterprise; this book should be too.

   | Kevin Korfmann
   | Philadelphia, 2026

   .. raw:: html

      <hr>

Welcome to *The Watchmaker's Guide to Population Genetics*.

**The Watchmaker's Guide** is a hands-on, build-it-yourself guide to the algorithms behind modern
population genetics. Every concept is explained from first principles, every method
implemented from scratch in Python, every abstraction earned rather than assumed.
Think of it as an apprenticeship: you start with raw materials -- basic math, simple
code -- and end up with working mechanisms that you built with your own hands.

By the end you won't just *run* the tools -- you'll know how to *build* them. And like
a watchmaker, once you've built something by hand, you can fix it, modify it, and
trust it completely.

----

.. toctree::
   :maxdepth: 3
   :caption: The Workshop

   philosophy

.. toctree::
   :maxdepth: 3
   :caption: The Workbench (Prerequisites)

   prerequisites/index

.. toctree::
   :maxdepth: 3
   :caption: Timepieces

   timepieces/index


How to Use This Guide
=====================

This book is organized around a central metaphor: **the watchmaker's workshop**.

Each **Timepiece** is a complete algorithm from population genetics -- a working
mechanism that you will disassemble, understand gear by gear, and then reassemble with
your own hands. Before diving into a Timepiece, you may need tools from
**The Workbench** -- prerequisite concepts explained with the same depth and care.

The recommended approach:

1. Read the **Philosophy** to understand why we do things this way.
2. Work through **The Workbench** to build your mathematical and computational toolkit.
3. Pick a **Timepiece** and build it, gear by gear.

Every chapter follows the same pattern:

- **Why** -- motivation and biological context (why does this mechanism exist?)
- **The Math** -- rigorous derivation from first principles (every step shown, every assumption stated)
- **The Code** -- Python implementation you write yourself (clear, readable, tested)
- **Verify** -- tests and sanity checks to confirm your gears mesh properly

We believe this rhythm -- motivation, math, code, verification -- is the most reliable
way to build genuine understanding. Each cycle reinforces the last, like the oscillation
of a balance wheel keeping perfect time.

.. admonition:: What you need

   - **Python 3.8+** with NumPy and SciPy (we'll explain every function we use)
   - **Some familiarity with probability and calculus** -- but don't worry if you're
     rusty. We introduce every concept we use, with intuition first and formulas second.
     If you know what a derivative is and what "probability" means, you have enough to
     start. We'll teach the rest as we go.
   - **Willingness to build things from scratch** -- this is the most important ingredient

.. admonition:: What you do NOT need

   - Prior knowledge of population genetics (we start from zero)
   - Experience with any existing genetics software
   - Advanced mathematics (we explain every concept as it arises -- exponential
     distributions, integrals, Markov chains, all of it)
   - A PhD (though you might feel like you've earned one by the end)

.. admonition:: A note on our teaching approach

   Many textbooks assume you already know probability, calculus, and linear algebra
   fluently. We don't. When we use the exponential distribution for the first time,
   we'll explain what it is and why it appears. When we integrate a function, we'll
   say what integration means in that context. When we invoke a matrix, we'll explain
   what it represents.

   This doesn't mean we skip the math -- we embrace it. It means we treat every
   mathematical tool like a physical tool on the workbench: we pick it up, show you
   what it does, demonstrate how to use it, and *then* put it to work.

   Like a good watchmaker teaching an apprentice, we never hand you a tool without
   first explaining what it's for.


.. bibliography::
   :all:
