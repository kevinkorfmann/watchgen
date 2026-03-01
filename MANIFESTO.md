# A Manifesto for Open Watchmaking

## On the State of Our Craft

Population genetics has produced some of the most elegant algorithms in computational biology. Methods for inferring ancestral recombination graphs, estimating demographic histories, dating genealogies, detecting selection. Each one a small marvel of mathematical engineering. Each one, increasingly, a black box.

The field has a reproducibility problem, but it is not the usual one. The code is often available. The papers are published. The data can sometimes be obtained. What is *not* reproducible is **understanding**. A graduate student can download PSMC and run it in an afternoon. But ask that student to explain *why* the transition matrix has the form it does, *how* the discretization of time interacts with the Baum-Welch updates, or *what* assumptions are violated when the method is applied to structured populations -- and you will find that the black box has done its work. The tool runs, but the knowledge of how it runs has not transferred.

This is not the fault of the original authors. These methods were born as research contributions, not pedagogical ones. The papers are dense because the ideas are dense. The supplements are long because the derivations are long. The implementations are in C++ because they need to be fast. None of this is wrong. But the cumulative effect is a field where the distance between "user" and "understander" grows wider with every new method.

**The Watchmaker's Guide to Population Genetics** is an attempt to close that gap.

## What This Project Is

This is a book -- open-source, freely available -- that rebuilds fifteen major population genetics algorithms from first principles. PSMC, SMC++, msprime, ARGweaver, tsinfer, SINGER, Threads, tsdate, Li & Stephens, moments, dadi, momi2, Gamma-SMC, PHLASH, CLUES. Every derivation shown. Every formula implemented in Python. Every implementation tested -- over a thousand unit tests verify that the gears mesh correctly.

The watchmaker metaphor is not decorative. It encodes a conviction: that the only reliable path to understanding a mechanism is to build it yourself, piece by piece, and verify that it works. We disassemble each algorithm into its smallest components, explain each one, and then reassemble the whole. Nothing is hidden behind "it can be shown" or "see supplementary materials." Nothing is assumed. Nothing is a black box.

But here is the critical point: **this project is not finished, and it cannot be finished by one person.**

## An Honest Admission

A single author -- or even a small team -- cannot be expert in every method covered here. The derivations have been checked, the code has been tested, the results have been compared against reference implementations. But population genetics is vast, the methods are subtle, and errors are not merely possible but probable. A misunderstood assumption here. A derivation that takes an unjustified shortcut there. An implementation that handles a boundary case incorrectly. A biological interpretation that oversimplifies.

**This is why the project is open source, and this is why we are asking for your help.**

We are not publishing a finished textbook and declaring it correct. We are releasing a working draft -- a mechanism assembled to the best of our ability -- and asking the community to inspect every gear. To find the flaws. To tighten the tolerances. To tell us where the escapement doesn't quite regulate, where a derivation skips a step that matters, where the code is correct but the explanation misleads.

This is not false modesty. It is an engineering reality. Complex mechanisms need many eyes. A single watchmaker builds the prototype; a guild of watchmakers certifies it keeps time.

## An Invitation

If you are a researcher who developed or contributed to one of these methods: **we want your scrutiny most of all.** No one understands ARGweaver like the people who built it. No one can validate our treatment of the moments equations like the team that derived them. If we have misrepresented your work, we want to know. If we have gotten it right, we want your confirmation. If you see a better way to explain something, we want your words.

If you are a student or postdoc working through these methods for the first time: **your confusion is data.** If a derivation loses you, that is a bug in our explanation, not a failure of your understanding. Open an issue. Tell us where you got stuck. The best teaching is shaped by the struggles of learners, not the assumptions of experts.

If you are an educator who teaches population genetics or computational biology: **use this freely, and tell us what's missing.** Adapt chapters for your courses. Assign the exercises. And when your students find errors -- they will -- send them our way.

If you are a developer who works on these tools: **check our implementations.** We have written educational Python that mirrors the logic of your production code. Where we have diverged, it may be a simplification -- or it may be a mistake. You are the ones who know the difference.

## The Principle

Science should not merely be open. It should be *accessible*. Publishing a paper is not enough if nobody outside a small circle can reconstruct the reasoning. Releasing code is not enough if the code is inscrutable. Open science, in its fullest sense, means opening not just the results but the *understanding* -- making it possible for anyone with curiosity and persistence to follow every step from first principles to working implementation.

This is the spirit of the watchmaker's workshop. The door is open. The tools are on the bench. The parts are laid out. You are welcome to pick them up, examine them, and tell us if they are machined correctly.

We believe that population genetics is too important -- and its methods too beautiful -- to remain locked inside black boxes. The algorithms that reconstruct our species' history, that detect the fingerprints of natural selection, that infer the demographic forces shaping genomes across millions of years -- these deserve to be understood, not just used. They deserve to be *taught*, not just cited.

## How to Contribute

The project lives on GitHub. Every chapter is a reStructuredText file. Every implementation is tested. Every derivation can be checked.

- **Found an error?** Open an issue. Be specific. Point to the equation, the line of code, the sentence that's wrong.
- **Want to improve an explanation?** Open a pull request. Better words are always welcome.
- **Want to add a new Timepiece?** Reach out. The collection is not closed.
- **Want to validate a chapter against the original method?** This is perhaps the most valuable contribution of all. A chapter that has been reviewed by the method's creators carries a credibility that no amount of self-checking can provide.

## A Final Word

The best watches in the world are not built in secret. They are built in workshops with open doors, by craftspeople who share techniques, who critique each other's work, who believe that the craft advances when knowledge flows freely.

We have laid out the parts on the bench. Come inspect the gears. Tell us which ones need recutting. Help us build something that keeps perfect time.

*The workshop is open.*
