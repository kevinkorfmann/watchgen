.. _philosophy:

==============================
The Watchmaker's Philosophy
==============================

Why Build It Yourself?
======================

Population genetics has a problem. The field's most important algorithms are locked
inside software packages that few people truly understand. Researchers run tools,
get numbers, and publish papers -- but ask them to explain what happens inside the
black box, and you'll get a shrug or a hand-wave toward a 40-page supplement.

This isn't anyone's fault. The methods *are* complex. The math *is* deep. The
implementations *are* thousands of lines of optimized code. It's rational to treat
them as black boxes.

But it's also dangerous.

When you don't understand the mechanism, you can't:

- Diagnose when results are wrong
- Adapt methods to new problems
- Know which assumptions are violated by your data
- Improve upon existing approaches
- Teach the next generation how the tools actually work

Imagine a watchmaker who can sell watches but cannot open one. Who can read the time
but cannot explain why the hands move. That's the state of much of computational
population genetics today: we read the output, but we can't explain the mechanism.

The Watchmaker's Way
====================

A watchmaker who only buys watches is a watch *dealer*, not a watchmaker. The craft
is in the building. The understanding is in the assembly. And the confidence -- the
ability to trust your results, diagnose problems, and push the boundaries of what's
possible -- comes only from having built the mechanism yourself, gear by gear.

In this guide, every algorithm is a **Timepiece** -- a mechanism you disassemble,
understand, and reassemble with your own hands. We don't skip steps. We don't
hand-wave. We don't say "it can be shown that..." and move on.

Here's what we promise:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - We WILL
     - We will NOT
   * - Derive every equation
     - Skip "obvious" steps
   * - Implement every algorithm
     - Use library functions as black boxes
   * - Explain every assumption
     - Hide behind "standard results"
   * - Test every implementation
     - Trust code without verification
   * - Build intuition first
     - Start with the most general case
   * - Teach the math as we go
     - Assume you already know everything

The Gears of Understanding
==========================

Every watch, no matter how complex, is built from smaller parts: gears, springs,
escapements, jewel bearings. Individually, each part is simple enough to hold in your
hand and understand completely. The complexity of a watch arises not from any single
part, but from how the parts mesh together.

The same is true for algorithms.

Each Timepiece is built from **gears** -- smaller mechanisms that mesh together.
We build the smallest gears first, test them, then combine them into larger
mechanisms. This mirrors how real algorithms are designed, and how understanding
actually develops: from the simple to the complex, with each step resting firmly on
the one before it.

We organize these gears into four categories, borrowing from horology:

1. **Escapement** -- The core mathematical insight that makes the whole thing tick.
   In a watch, the escapement regulates the release of energy. In our algorithms,
   this is the key formula or theorem that everything else depends on.

2. **Gear Train** -- The computational machinery that turns insight into algorithm.
   In a watch, the gear train transmits energy from the mainspring to the hands.
   Here, it's the sequence of computational steps that transforms the core insight
   into a working procedure.

3. **Mainspring** -- The data structures and implementation details that store energy
   and state. In a watch, the mainspring stores the energy that drives everything.
   In code, these are the arrays, tables, and structures that hold the data the
   algorithm works on.

4. **Case and Dial** -- The interface: inputs, outputs, and interpretation. The case
   protects the mechanism; the dial presents the result. For us, this is how the
   algorithm connects to real data and how we interpret what it tells us.

You wouldn't try to understand a watch by staring at the finished product through
the crystal. You'd take it apart, lay out every piece on the bench, understand what
each does, and carefully put it back together. That's exactly what we do here.

On Mathematical Rigor
=====================

We don't shy away from math -- we embrace it. But we also believe that mathematics
is a *tool*, not a gatekeeping mechanism. Every formula in this book is here because
it helps you understand something, not because it makes us look clever.

Our approach to mathematics:

- **Every equation is motivated before it's stated.** We always explain *why* we need
  a formula before writing it down. What question does it answer? What would we be
  missing without it?

- **Every derivation is accompanied by intuition.** Alongside the formal steps, we
  provide a plain-English narrative: "this term counts the probability of survival,
  and this term counts the probability of the event happening right now."

- **Every formula is implemented in code so you can see it work.** Mathematics on paper
  can feel abstract. Mathematics in Python, producing numbers you can check, feels
  concrete and real. Code is our way of testing understanding: if you can implement a
  formula correctly, you understand it.

- **Notation is consistent and explicitly defined.** We use the same symbols for the
  same concepts throughout the book, and we define every symbol when it first appears.

When you encounter a formula in this book, you should be able to:

1. Say in plain English what it computes
2. Explain why it has the form it does
3. Implement it in Python
4. Verify it produces correct results on simple examples

If any of these fail, we've failed as authors. Like a watch that doesn't keep time,
a formula you can't use is a formula that isn't working.

On Teaching Probability and Calculus
=====================================

This book uses probability and calculus extensively -- they are essential tools for
understanding how genetic algorithms work. But we don't assume you arrive with these
tools already sharp and polished in your toolkit.

Here's our approach:

**For probability:** We introduce each concept as it naturally arises. The first time
we need the exponential distribution, we explain what it is, where it comes from, why
it appears in our context, and what its parameters mean. The same for Bayes' theorem,
conditional probability, Markov chains, and every other concept. We provide
"Probability Aside" boxes that give you the background you need, right when you need it.

**For calculus:** We use derivatives and integrals, but we always explain what they
mean in context. A derivative is a rate of change -- we'll say what's changing and
how fast. An integral is a sum -- we'll say what's being summed and why. When we
differentiate or integrate, we show the steps and explain the rules being used.

**For Python:** We explain every non-obvious construct the first time it appears.
If you know basic Python (variables, loops, functions), you have enough to start.
We'll explain NumPy array operations, list comprehensions, and anything else as we go.

Think of it this way: a master watchmaker teaching an apprentice doesn't assume the
apprentice already knows metallurgy, thermodynamics, and materials science. Instead,
the master teaches these things *as they become relevant* -- explaining the properties
of steel when it's time to shape a spring, explaining thermal expansion when it's time
to fit a jewel bearing. Context makes learning stick.

That's our philosophy: teach every tool in the context where it's needed, so the
knowledge arrives precisely when it's useful.

On Python Implementations
=========================

We use Python because it reads like pseudocode. Our implementations prioritize
**clarity over performance**. Real production tools use C++ for speed -- but you
can't learn from code you can't read.

Every implementation follows these rules:

- **No magic imports.** We use NumPy and SciPy, nothing else. Every helper
  function is written from scratch in front of you. When we call a NumPy function
  for the first time, we explain what it does.
- **No premature optimization.** We write the naive version first, understand it,
  then optimize only where the math demands it. The naive version often reveals the
  structure of the algorithm more clearly than the optimized one.
- **No hidden state.** Every variable is named descriptively, every parameter is
  documented. You should be able to read the code like a recipe -- each line
  corresponding to a step in the mathematical derivation.
- **Every code block can be run.** The code in this book is not pseudocode dressed
  up to look like Python. It runs. It produces the numbers we claim. You can type
  it into a notebook and verify for yourself.

.. admonition:: A note on production code

   The code in this book is *educational* code. It is correct, tested, and
   complete -- but it is not optimized for production use. Real tools like
   SINGER, Relate, and tsinfer use C++ with careful memory management for
   good reason. Our Python implementations exist to help you understand the
   algorithms, not to replace the production tools.

   Think of the difference like this: a watchmaking student first builds a
   movement from brass using hand tools. Later, they may work with modern
   CNC machines and exotic alloys. But the hand-built movement teaches them
   something that no factory ever could -- the deep understanding of *why*
   each part has the shape it does.

Your Journey
============

You're about to build algorithms that took decades to develop. You'll derive
equations that fill supplementary materials. You'll implement methods that
exist as thousands of lines of C++.

It won't be easy. But neither is watchmaking. The reward, in both cases, is the
same: the profound satisfaction of understanding a complex mechanism so completely
that you could build it again from memory.

When you're done, you'll own the knowledge completely -- no black boxes, no
hand-waving, no mystery, just gears you built yourself, ticking reliably,
because you understand every one.

*Let's open the case and get to work.*
