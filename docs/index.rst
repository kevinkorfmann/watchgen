.. _index:

====================================================
The Watchmaker's Guide to Population Genetics
====================================================

   *A watchmaker doesn't just use a watch -- they understand every gear, every spring, every mechanism. Nothing is a black box.*

.. image:: https://img.shields.io/badge/philosophy-build%20it%20yourself-blue
   :alt: Build It Yourself

----

Welcome to *The Watchmaker's Guide to Population Genetics*.

This book exists because population genetics is full of black boxes. Powerful tools
that produce results nobody fully understands anymore. Methods inherited from papers,
wrapped in software, used without question. Researchers run programs, get numbers,
and cite supplementary materials they've never read.

We think there's a better way.

**The Watchmaker's Guide** is a hands-on, build-it-yourself guide to the algorithms behind modern
population genetics. Every concept is explained from first principles, every method
implemented from scratch in Python, every abstraction earned rather than assumed.
Think of it as an apprenticeship: you start with raw materials -- basic math, simple
code -- and end up with working mechanisms that you built with your own hands.

By the end you won't just *run* the tools -- you'll know how to *build* them. And like
a watchmaker, once you've built something by hand, you can fix it, modify it, and
trust it completely.

**The math is here. The code is here. The exercises are here. You don't need to go anywhere else.**

----

.. toctree::
   :maxdepth: 2
   :caption: The Workshop

   philosophy

.. toctree::
   :maxdepth: 2
   :caption: The Workbench (Prerequisites)

   prerequisites/index

.. toctree::
   :maxdepth: 2
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
