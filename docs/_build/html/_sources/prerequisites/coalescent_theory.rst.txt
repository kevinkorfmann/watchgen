.. _coalescent_theory:

==================
Coalescent Theory
==================

   *To understand the present, you must run the clock backwards.*

The Big Idea
============

Most of biology thinks forward in time: mutations arise, organisms reproduce,
populations evolve. But in population genetics, some of the most powerful insights
come from thinking **backwards**.

Here's the intuition. Imagine you've collected DNA from several individuals in a
population. Each of those DNA sequences had a "parent" sequence in the previous
generation, and that parent had a parent, and so on. If you trace these lineages
backwards through time, something remarkable happens: they converge. Lineages that
were separate in the present share a common ancestor in the past. Eventually, if you
go back far enough, *all* the lineages trace back to a single ancestor.

This process of lineages merging as we look backwards is called **coalescence**
(from "to coalesce" -- to come together, to merge). The mathematical theory that
describes it is called **coalescent theory**, and it is the escapement mechanism at
the heart of every Timepiece in this book.

.. admonition:: Why "escapement"?

   In a mechanical watch, the escapement is the component that regulates the release
   of energy from the mainspring. It's what makes the watch *tick* -- the fundamental
   oscillation that everything else depends on. Coalescent theory plays the same role
   in population genetics: it's the mathematical heartbeat that drives every algorithm
   we'll build.

Why think backwards? Because it's far more efficient. A population might contain
millions of individuals, but we've only sampled a handful. Going forward, we'd need
to simulate the entire population. Going backwards, we only track the lineages of our
actual samples -- and those lineages simplify quickly as they merge.

The Wright-Fisher Model (Forward in Time)
==========================================

Before going backwards, let's establish the forward-time model that the coalescent
approximates. This model is deliberately simple -- it strips away all the biological
complexity (mating, geography, natural selection) to capture just the essence of
random inheritance.

Consider a population of :math:`N` diploid individuals. "Diploid" means each
individual carries two copies of each chromosome -- one inherited from each parent.
Since we're tracking a single genetic locus (a specific position on the genome), there
are :math:`2N` gene copies in the population.

.. admonition:: Why :math:`2N`?

   Diploid organisms (including humans) carry two copies of each chromosome. When we
   track a single locus, there are :math:`2N` copies in the population -- two per
   individual. We often write :math:`N_e` for the **effective population size** (which
   accounts for various complications like unequal sex ratios or fluctuating population
   size), and the total number of gene copies is :math:`2N_e`. For now, just think of
   :math:`2N` as "the total number of gene copies we're tracking."

In the **Wright-Fisher model**, reproduction works like this:

1. **Generations are discrete and non-overlapping.** The entire population is replaced
   at once -- all parents die, all offspring are born simultaneously.

2. **Each gene copy in the next generation chooses its parent uniformly at random**
   from the :math:`2N` copies in the current generation. "Uniformly at random" means
   every gene copy in the parental generation has an equal chance :math:`1/(2N)` of
   being chosen as the parent of any given offspring copy.

3. **The population size is constant** -- there are always exactly :math:`2N` gene
   copies.

This is obviously not how real biology works. But its simplicity is its strength:
it captures the essential randomness of inheritance (which gene copy gets passed on
is largely a matter of chance), and many of the conclusions we draw from it are
remarkably robust to the simplifying assumptions.

Let's simulate this model in Python. If you're new to NumPy, don't worry -- we'll
explain each function as we use it.

.. code-block:: python

   import numpy as np

   def wright_fisher_forward(two_N, n_generations):
       """Simulate the Wright-Fisher model forward in time.

       Parameters
       ----------
       two_N : int
           Total number of gene copies (2 * diploid population size).
       n_generations : int
           Number of generations to simulate.

       Returns
       -------
       parent_table : ndarray of shape (n_generations, two_N)
           parent_table[g, i] = parent index of individual i in generation g.
       """
       # np.zeros creates an array filled with zeros.
       # dtype=int means the entries are integers (parent indices).
       parent_table = np.zeros((n_generations, two_N), dtype=int)

       for g in range(n_generations):
           # np.random.randint(0, two_N, size=two_N) generates 'two_N'
           # random integers, each between 0 and two_N-1 (inclusive).
           # This is the "each gene copy picks a random parent" step.
           parent_table[g] = np.random.randint(0, two_N, size=two_N)

       return parent_table

   # Try it: 10 gene copies, 20 generations
   # np.random.seed(42) makes the random numbers reproducible --
   # you'll get the same result every time you run this code.
   np.random.seed(42)
   parents = wright_fisher_forward(10, 20)
   print("Parent of each individual in generation 0 (most recent):")
   print(parents[0])

Notice what the code does: in each generation, every gene copy independently picks a
random parent from the previous generation. Some parents get picked multiple times
(their genes are passed on to multiple offspring), while others get picked zero times
(their lineage dies out). This randomness is the engine that drives genetic drift.

Going Backwards: The Coalescent
=================================

Now the key insight. Instead of tracking the entire population forward, we
**sample** :math:`n` lineages and trace them **backwards**. At each generation,
two lineages that chose the same parent **coalesce** into one.

This is like tracing the hands of a clock backwards: the minute and hour hands
start at different positions (our sampled lineages) and, as we rewind, we watch
for the moments when they align (coalescence events). Each alignment reduces the
number of independent lineages by one, until finally there's just a single hand
pointing to the most recent common ancestor.

The probability that two specific lineages coalesce in a given generation
-------------------------------------------------------------------------------------------

Let's work out the most basic question: if we pick two specific lineages from our
sample, what is the probability that they share a parent in the immediately preceding
generation?

In a population of :math:`2N` gene copies, consider two specific lineages. Each one
independently chooses a parent uniformly at random from the :math:`2N` copies in the
previous generation.

The first lineage picks some parent -- it doesn't matter which. The second
lineage picks each of the :math:`2N` possible parents with equal probability
:math:`1/(2N)`. It picks the *same* parent as the first lineage with probability:

.. math::

   P(\text{coalesce in one generation}) = \frac{1}{2N}

and a *different* parent with probability:

.. math::

   P(\text{do not coalesce}) = 1 - \frac{1}{2N} = \frac{2N - 1}{2N}

.. admonition:: Why uniform and independent?

   This follows directly from the Wright-Fisher model: each gene copy in the
   next generation picks its parent *independently* and *uniformly* from the
   :math:`2N` copies. The "uniformly" gives us :math:`1/(2N)` and the
   "independently" lets us treat the two lineages separately. For large :math:`N`,
   sampling with replacement (WF model) and without replacement are nearly
   identical -- the error is :math:`O(1/N^2)`.

For human populations where :math:`N_e \approx 10{,}000`, we get :math:`2N = 20{,}000`,
so the probability of coalescence in any given generation is just :math:`1/20{,}000 =
0.00005`. That's tiny! This means that for most generations, nothing happens -- the
two lineages just independently trace back to different parents. But over thousands of
generations, coalescence becomes inevitable.

Waiting time to coalescence
----------------------------

How many generations do we wait until two lineages coalesce? Let's think about this
carefully, because it introduces an important probability distribution.

For coalescence to happen for the *first time* at generation :math:`t`, two things
must be true:

1. The lineages did **not** coalesce in each of the first :math:`t-1` generations
2. They **did** coalesce in generation :math:`t`

Since each generation is independent (the Wright-Fisher model has no memory), we can
multiply the probabilities:

.. math::

   P(T_2 = t) = \underbrace{\left(1 - \frac{1}{2N}\right)^{t-1}}_{\text{survived } t-1 \text{ generations}} \cdot \underbrace{\frac{1}{2N}}_{\text{coalesced at } t}

.. admonition:: Probability aside: the geometric distribution

   This is a **geometric distribution** with success probability :math:`p = 1/(2N)`.
   In general, if you repeat an experiment independently until the first success,
   and the probability of success on each trial is :math:`p`, then the number of
   trials until the first success follows a geometric distribution:

   .. math::

      P(T = t) = (1-p)^{t-1} \cdot p, \quad t = 1, 2, 3, \ldots

   Its mean (expected value) is :math:`E[T] = 1/p`. The intuition: if each trial
   succeeds with probability :math:`p`, you expect to need about :math:`1/p` trials.
   For our coalescent, :math:`p = 1/(2N)`, so the expected waiting time is :math:`2N`
   generations. In a population of 10,000 diploid individuals, two lineages take an
   average of 20,000 generations to coalesce.

For large :math:`N`, this geometric distribution is well approximated by something
smoother and easier to work with: the **exponential distribution**. Let's derive this
carefully, because the exponential distribution will appear in nearly every chapter of
this book.

**The geometric-to-exponential limit.** We want to show that as :math:`N \to \infty`,
the geometric distribution converges to an exponential distribution when time is
measured in units of :math:`2N` generations.

The key is to **rescale time**. Instead of counting individual generations, we measure
time in units of :math:`2N` generations. Define:

.. math::

   \tau = \frac{t}{2N}

So :math:`\tau = 1` corresponds to :math:`2N` generations, :math:`\tau = 0.5`
corresponds to :math:`N` generations, and so on.

Now, what is the probability that two lineages have *not* coalesced after :math:`t = 2N\tau`
generations?

.. math::

   P(T_2 > t) = \left(1 - \frac{1}{2N}\right)^{t} = \left(1 - \frac{1}{2N}\right)^{2N\tau}

Here we use a beautiful fact from calculus:

.. admonition:: Calculus aside: the exponential limit

   One of the most important limits in mathematics is:

   .. math::

      \lim_{m \to \infty} \left(1 - \frac{1}{m}\right)^m = e^{-1} \approx 0.3679

   More generally, for any constant :math:`c`:

   .. math::

      \lim_{m \to \infty} \left(1 - \frac{c}{m}\right)^m = e^{-c}

   This limit connects discrete compound processes to the exponential function. It
   says: if something small (:math:`1/m`) happens repeatedly (:math:`m` times),
   the combined effect is exponential.

Setting :math:`m = 2N` and using this limit:

.. math::

   \left(1 - \frac{1}{2N}\right)^{2N\tau}
   = \left[\left(1 - \frac{1}{2N}\right)^{2N}\right]^\tau
   \xrightarrow{N \to \infty} \left[e^{-1}\right]^\tau = e^{-\tau}

Therefore, in the limit:

.. math::

   P(T_2 / (2N) > \tau) \to e^{-\tau}

.. admonition:: Probability aside: the exponential distribution

   A random variable :math:`X` follows an **exponential distribution** with rate
   :math:`\lambda` (written :math:`X \sim \text{Exp}(\lambda)`) if:

   .. math::

      P(X > x) = e^{-\lambda x} \quad \text{for } x \geq 0

   Key properties:

   - **Mean**: :math:`E[X] = 1/\lambda`
   - **Variance**: :math:`\text{Var}(X) = 1/\lambda^2`
   - **Memoryless property**: :math:`P(X > s + t \mid X > s) = P(X > t)`. Knowing
     you've already waited :math:`s` units doesn't change the distribution of
     additional waiting time. This is what makes the exponential distribution the
     continuous analogue of the geometric.
   - **Density**: :math:`f(x) = \lambda e^{-\lambda x}` (obtained by differentiating
     :math:`-P(X > x)`)

   The "rate" :math:`\lambda` controls how quickly events happen. Higher rate = shorter
   waits. Rate 1 means "on average, one event per unit time."

   Since :math:`P(T_2/(2N) > \tau) \to e^{-\tau}`, we have:

   .. math::

      T_2 / (2N) \xrightarrow{d} \text{Exp}(1) \quad \text{as } N \to \infty

This tells us: the coalescence time of two lineages, measured in units of :math:`2N`
generations, is approximately exponentially distributed with rate 1.

.. admonition:: Why scale by :math:`2N`?

   The probability of coalescence per generation is :math:`1/(2N)`. The expected
   waiting time is :math:`\mathbb{E}[T_2] = 2N` generations (the mean of a
   geometric with parameter :math:`p` is :math:`1/p`). After rescaling by
   :math:`2N`, the expected coalescence time becomes :math:`2N/(2N) = 1`,
   matching the mean of :math:`\text{Exp}(1)`. This rescaling is the natural
   "clock speed" of the coalescent -- it makes the coalescence rate for a pair
   exactly 1, which simplifies all subsequent mathematics enormously.

   From now on, unless stated otherwise, **all times in this book are measured in
   coalescent time units** (multiples of :math:`2N` generations).

.. admonition:: How good is the approximation?

   The geometric and exponential distributions agree closely even for moderate
   :math:`N`. For :math:`2N = 100`, the relative error in the survival function
   is less than 1% for all :math:`\tau`. For :math:`2N = 1000`, it's less than 0.1%.
   In practice, effective population sizes are in the thousands to millions, so
   the approximation is excellent.

Let's verify this with simulation. We'll directly simulate pairs of lineages picking
random parents until they coalesce, repeat many times, and check that the distribution
of waiting times matches the exponential.

.. code-block:: python

   def coalescence_time_two_lineages(two_N, n_replicates=10000):
       """Simulate coalescence times for pairs of lineages.

       Each replicate: two lineages independently pick random parents
       each generation until they pick the same one (coalescence).

       Parameters
       ----------
       two_N : int
           Total number of gene copies (2N).
       n_replicates : int
           Number of independent simulations to run.

       Returns
       -------
       times : ndarray
           Coalescence times in generations.
       """
       times = np.zeros(n_replicates)
       for rep in range(n_replicates):
           t = 0
           while True:
               t += 1
               # Both lineages pick a parent at random
               parent1 = np.random.randint(0, two_N)
               parent2 = np.random.randint(0, two_N)
               if parent1 == parent2:
                   break
           times[rep] = t
       return times

   two_N = 100
   times = coalescence_time_two_lineages(two_N, n_replicates=50000)

   # Convert to coalescent time units (divide by 2N)
   scaled_times = times / two_N

   print(f"Mean coalescence time (in 2N generations): {scaled_times.mean():.3f}")
   print(f"Expected (Exp(1) mean):                    1.000")
   print(f"Variance:                                   {scaled_times.var():.3f}")
   print(f"Expected (Exp(1) variance):                 1.000")

You should see the mean and variance very close to 1.000, confirming that the
exponential approximation works well even for :math:`2N = 100`.


The Coalescent with :math:`n` Samples
======================================

So far we've considered just two lineages. What happens with :math:`n` samples?

The key insight is that coalescence events happen between **pairs** of lineages, and
each pair behaves independently. When there are :math:`k` lineages remaining, we
need to count how many pairs there are and figure out how quickly one of them coalesces.

**Counting pairs.** The number of distinct pairs you can form from :math:`k` items
is the **binomial coefficient**:

.. math::

   \binom{k}{2} = \frac{k!}{2!(k-2)!} = \frac{k(k-1)}{2}

.. admonition:: Why :math:`k(k-1)/2`?

   To choose a pair from :math:`k` items: pick the first item (:math:`k` choices),
   then pick the second (:math:`k-1` choices). But order doesn't matter (the pair
   {A, B} is the same as {B, A}), so divide by 2. For :math:`k = 5`:
   :math:`\binom{5}{2} = 5 \times 4 / 2 = 10` pairs.

Since each pair coalesces independently at rate 1 (in coalescent time units), and
there are :math:`\binom{k}{2}` pairs, the **total coalescence rate** is:

.. math::

   \lambda_k = \binom{k}{2} = \frac{k(k-1)}{2}

.. admonition:: Probability aside: rates of competing exponentials

   When multiple independent exponential events are "racing" to happen first, the
   time until the first event is also exponential, with a rate equal to the **sum**
   of the individual rates. This is a fundamental property:

   If :math:`X_1 \sim \text{Exp}(\lambda_1)` and :math:`X_2 \sim \text{Exp}(\lambda_2)`
   are independent, then :math:`\min(X_1, X_2) \sim \text{Exp}(\lambda_1 + \lambda_2)`.

   With :math:`\binom{k}{2}` independent pairs, each with rate 1, the first
   coalescence happens at rate :math:`\binom{k}{2}`.

The waiting time until the next coalescence event (any pair) is:

.. math::

   T_k \sim \text{Exp}\left(\binom{k}{2}\right)

This means the expected waiting time is :math:`E[T_k] = 2/(k(k-1))`. Notice how this
decreases rapidly as :math:`k` grows: with many lineages, coalescence events happen
quickly because there are so many possible pairs. Like the ticking of a clock that
gradually slows down -- the coalescent starts with rapid events and progressively
decelerates.

When a coalescence event does occur, the pair that coalesces is chosen uniformly at
random from all :math:`\binom{k}{2}` pairs (because all pairs are racing at the same
rate, each is equally likely to win).

**The total tree height.** The time to the most recent common ancestor (MRCA) is
the sum of all the waiting times:

.. math::

   T_{\text{MRCA}} = \sum_{k=2}^{n} T_k

and its expectation is:

.. math::

   \mathbb{E}[T_{\text{MRCA}}] = \sum_{k=2}^{n} \mathbb{E}[T_k]
   = \sum_{k=2}^{n} \frac{2}{k(k-1)}

Let's evaluate this sum. The technique is called **partial fractions** -- a way to
split a complicated fraction into simpler pieces that cancel nicely.

We decompose:

.. math::

   \frac{1}{k(k-1)} = \frac{A}{k-1} + \frac{B}{k}

Multiplying both sides by :math:`k(k-1)`:

.. math::

   1 = Ak + B(k-1)

Setting :math:`k = 0`: :math:`1 = -B`, so :math:`B = -1`.
Setting :math:`k = 1`: :math:`1 = A`, so :math:`A = 1`. Therefore:

.. math::

   \frac{1}{k(k-1)} = \frac{1}{k-1} - \frac{1}{k}

This is a **telescoping sum** -- a sum where most terms cancel in pairs, like
interlocking gears where each tooth advances exactly as far as the previous tooth
retreats. Substituting:

.. math::

   \sum_{k=2}^{n} \frac{2}{k(k-1)} &= 2\sum_{k=2}^{n}\left(\frac{1}{k-1} - \frac{1}{k}\right) \\
   &= 2\left[\left(\frac{1}{1} - \frac{1}{2}\right) + \left(\frac{1}{2} - \frac{1}{3}\right)
   + \cdots + \left(\frac{1}{n-1} - \frac{1}{n}\right)\right] \\
   &= 2\left(\frac{1}{1} - \frac{1}{n}\right) = 2\left(1 - \frac{1}{n}\right)

Most terms cancel in pairs: the :math:`-1/2` from the first term cancels the
:math:`+1/2` from the second, the :math:`-1/3` from the second cancels the
:math:`+1/3` from the third, and so on. Only the very first and very last terms
survive.

.. admonition:: Key insight: diminishing returns of sampling

   As :math:`n \to \infty`, the expected TMRCA converges to :math:`2(1 - 1/n) \to 2`
   (in coalescent time units). Adding more samples barely changes the tree
   height! Here are the numbers:

   - :math:`n = 2`: TMRCA = 1.000
   - :math:`n = 5`: TMRCA = 1.600
   - :math:`n = 10`: TMRCA = 1.800
   - :math:`n = 100`: TMRCA = 1.980
   - :math:`n = 1000`: TMRCA = 1.998

   The diminishing returns are dramatic. Most coalescent events happen quickly when
   there are many lineages (the rate :math:`k(k-1)/2` is huge), so the tree rapidly
   narrows to just a few lineages. The last few coalescences take most of the time.

   This is like the final adjustments in watchmaking: the bulk of the mechanism
   comes together quickly, but the last few precision alignments take as long as
   all the earlier work combined.

Let's implement the full coalescent simulation:

.. code-block:: python

   def simulate_coalescent(n):
       """Simulate a coalescent tree for n samples.

       This simulates Kingman's coalescent: starting with n lineages,
       we repeatedly wait an exponential time and merge a random pair,
       until only one lineage (the MRCA) remains.

       Parameters
       ----------
       n : int
           Number of samples (leaf nodes).

       Returns
       -------
       events : list of (time, child1, child2, parent)
           Coalescence events in chronological order (going back in time).
           'time' is in coalescent units (multiples of 2N generations).
       """
       # Active lineages, labeled 0 to n-1
       lineages = list(range(n))
       next_label = n  # labels for internal (ancestor) nodes
       events = []
       current_time = 0.0

       while len(lineages) > 1:
           k = len(lineages)
           rate = k * (k - 1) / 2  # binom(k, 2)

           # Sample waiting time from Exp(rate).
           # np.random.exponential(scale) draws from Exp with mean = scale,
           # so scale = 1/rate gives rate = rate.
           wait = np.random.exponential(1.0 / rate)
           current_time += wait

           # Choose a random pair to coalesce.
           # np.random.choice(n, size=2, replace=False) picks 2 distinct
           # indices from {0, 1, ..., n-1}.
           i, j = np.random.choice(len(lineages), size=2, replace=False)
           child1 = lineages[i]
           child2 = lineages[j]

           # Create parent node
           parent = next_label
           next_label += 1
           events.append((current_time, child1, child2, parent))

           # Remove children, add parent
           lineages = [l for idx, l in enumerate(lineages)
                       if idx != i and idx != j]
           lineages.append(parent)

       return events

   # Simulate and print
   np.random.seed(42)
   events = simulate_coalescent(5)
   print("Coalescence events (time, child1, child2, parent):")
   for t, c1, c2, p in events:
       print(f"  t={t:.4f}: lineages {c1} and {c2} -> {p}")
   print(f"\nTMRCA = {events[-1][0]:.4f}")
   print(f"Expected TMRCA = {2*(1 - 1/5):.4f}")

Read through the output carefully. You'll see that the first few coalescence events
happen quickly (when there are many lineages competing), while the final coalescence
to the MRCA takes longer. This matches the theory: the rate drops from
:math:`\binom{5}{2} = 10` down to :math:`\binom{2}{2} = 1`.

Expected Number of Lineages at Time :math:`t`
===============================================

A result that will be critical for the SINGER algorithm: given :math:`n` samples,
what is the expected number of lineages remaining at time :math:`t`?

Frost and Volz (2010) showed that for large :math:`n`, the number of lineages
:math:`\lambda(t)` is nearly deterministic and satisfies the **ordinary differential
equation (ODE)**:

.. math::

   \frac{d\lambda}{dt} = -\binom{\lambda}{2} = -\frac{\lambda(\lambda - 1)}{2}

with initial condition :math:`\lambda(0) = n`.

.. admonition:: Calculus aside: what is a differential equation?

   A differential equation relates a quantity to its rate of change. Here,
   :math:`d\lambda/dt` is the rate at which the number of lineages changes
   over time. The equation says: the rate of decrease equals the number of
   possible pairs, :math:`\binom{\lambda}{2}`. This makes sense because each
   coalescence (which reduces :math:`\lambda` by 1) happens at rate
   :math:`\binom{\lambda}{2}`. The minus sign indicates that :math:`\lambda`
   is decreasing.

   Solving a differential equation means finding a function :math:`\lambda(t)`
   that satisfies this relationship. We do this below using a technique called
   "separation of variables."

This is a **separable ODE**, meaning we can move all the :math:`\lambda` terms to one
side and all the :math:`t` terms to the other. Separating variables:

.. math::

   \frac{d\lambda}{\lambda(\lambda - 1)} = -\frac{dt}{2}

.. admonition:: Calculus aside: separation of variables

   This technique works when a differential equation has the form
   :math:`f(\lambda)\,d\lambda = g(t)\,dt`. We can integrate both sides
   independently. The left side becomes an integral in :math:`\lambda`,
   the right side an integral in :math:`t`.

Using partial fractions on the left (the same technique we used for the telescoping
sum -- :math:`\frac{1}{\lambda(\lambda-1)} = \frac{1}{\lambda-1} - \frac{1}{\lambda}`):

.. math::

   \left(\frac{1}{\lambda - 1} - \frac{1}{\lambda}\right) d\lambda = -\frac{dt}{2}

Integrating both sides:

.. math::

   \ln(\lambda - 1) - \ln(\lambda) = -\frac{t}{2} + C

.. admonition:: Calculus aside: the integral :math:`\int \frac{1}{x}\,dx = \ln|x|`

   The natural logarithm :math:`\ln(x)` is the function whose derivative is
   :math:`1/x`. Equivalently, :math:`\int \frac{1}{x}\,dx = \ln|x| + C`.
   This is one of the most important integrals in mathematics, and it appears
   throughout this book.

Using the logarithm rule :math:`\ln(a) - \ln(b) = \ln(a/b)`:

.. math::

   \ln\left(\frac{\lambda - 1}{\lambda}\right) = -\frac{t}{2} + C

At :math:`t = 0`, :math:`\lambda = n`, so :math:`C = \ln\left(\frac{n-1}{n}\right)`.
Exponentiating both sides (applying :math:`e^x` to undo :math:`\ln`):

.. math::

   \frac{\lambda - 1}{\lambda} = \frac{n-1}{n} e^{-t/2}

Solving for :math:`\lambda` (rewrite the left side as :math:`1 - 1/\lambda`):

.. math::

   1 - \frac{1}{\lambda} = \frac{n-1}{n} e^{-t/2}

.. math::

   \frac{1}{\lambda} = 1 - \frac{n-1}{n} e^{-t/2} = \frac{n - (n-1)e^{-t/2}}{n}

Inverting:

.. math::

   \lambda(t) = \frac{n}{n - (n-1)e^{-t/2}} = \frac{n}{n + (1 - n)e^{-t/2}}

where the last step uses :math:`-(n-1) = 1 - n`.

Let's verify this formula by comparing it to simulation:

.. code-block:: python

   def expected_lineages(t, n):
       """Expected number of lineages at time t for n initial samples.

       Uses the large-n deterministic approximation (Frost & Volz, 2010).

       Parameters
       ----------
       t : float
           Time in coalescent units.
       n : int
           Number of initial samples.

       Returns
       -------
       float
           Expected number of lineages at time t.
       """
       return n / (n + (1 - n) * np.exp(-t / 2))

   def simulate_lineage_count(n, t, n_replicates=10000):
       """Estimate E[lineages at time t] by simulation.

       Simulates the coalescent n_replicates times, counts how many
       lineages remain at time t, and returns the average.
       """
       counts = np.zeros(n_replicates)
       for rep in range(n_replicates):
           k = n
           current_time = 0.0
           while k > 1 and current_time < t:
               rate = k * (k - 1) / 2
               wait = np.random.exponential(1.0 / rate)
               if current_time + wait > t:
                   break
               current_time += wait
               k -= 1
           counts[rep] = k
       return counts.mean()

   n = 50
   for t in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
       approx = expected_lineages(t, n)
       simulated = simulate_lineage_count(n, t, n_replicates=5000)
       print(f"t={t:.2f}: approx={approx:.2f}, simulated={simulated:.2f}")

The agreement should be excellent, especially for :math:`n = 50`, confirming that the
deterministic approximation captures the coalescent dynamics well.

Mutations on the Coalescent Tree
==================================

So far, the coalescent tells us about the genealogical tree -- its shape and timing.
But what we actually *observe* in real data is not the tree itself, but **mutations**:
differences between DNA sequences at specific positions.

Mutations are added to the coalescent tree under the **infinite sites model**:

- Each mutation occurs at a unique position on the genome (no position mutates twice)
- Mutations arise as a **Poisson process** along each branch of the tree
- The rate is governed by :math:`\theta = 4N_e\mu`, where :math:`\mu` is the
  per-base-pair, per-generation mutation rate

.. admonition:: Where does :math:`\theta = 4N_e\mu` come from?

   In the Wright-Fisher model, each base pair mutates with probability :math:`\mu`
   per generation. A branch of length :math:`\ell` in coalescent time units
   corresponds to :math:`2N_e \cdot \ell` generations (since 1 coalescent unit =
   :math:`2N_e` generations). The expected number of mutations on this branch at a
   single site is:

   .. math::

      \mu \times 2N_e\ell = \frac{4N_e\mu}{2} \cdot \ell = \frac{\theta}{2} \cdot \ell

   The factor of :math:`4N_e` appears because :math:`\theta` is defined as
   :math:`4N_e\mu` (a convention for diploid organisms), and the :math:`1/2`
   appears because we're converting from the :math:`4N_e\mu` scaling to a
   per-unit-coalescent-time rate. The mutation rate **per coalescent time unit
   per base pair** is :math:`\theta/2`.

.. admonition:: Probability aside: the Poisson process

   A **Poisson process** with rate :math:`r` is a model for events that occur
   randomly and independently over time (or space). In a time interval of length
   :math:`L`, the number of events follows a **Poisson distribution** with
   parameter :math:`rL`:

   .. math::

      P(k \text{ events in } [0, L]) = \frac{(rL)^k}{k!} e^{-rL}

   Key properties:

   - **Mean number of events**: :math:`rL`
   - **Probability of zero events**: :math:`e^{-rL}` (the zeroth term of the sum)
   - **Probability of at least one event**: :math:`1 - e^{-rL}`
   - Events in non-overlapping intervals are independent

   This model is a natural limit of many rare independent trials. If each base pair
   has a tiny probability of mutating per generation, and there are many generations,
   the total mutation count converges to a Poisson distribution (this is the
   **Poisson limit theorem**).

The number of mutations on a branch of length :math:`\ell` follows a Poisson
distribution with mean :math:`\theta\ell/2`. The probability that a branch carries
**no** mutations at a single site is:

.. math::

   P(\text{no mutation}) = \frac{(\theta\ell/2)^0}{0!} e^{-\theta\ell/2} = \exp\left(-\frac{\theta}{2} \ell\right)

and the probability of **at least one** mutation is:

.. math::

   P(\text{mutation}) = 1 - \exp\left(-\frac{\theta}{2} \ell\right)

For small :math:`\theta\ell/2`, we can use the approximation :math:`e^{-x} \approx 1 - x`
(valid when :math:`x` is much less than 1):

.. math::

   P(\text{mutation}) \approx \frac{\theta}{2}\ell \quad \text{when } \frac{\theta}{2}\ell \ll 1

For human data, :math:`\mu \approx 1.25 \times 10^{-8}` per bp per generation
and :math:`N_e \approx 10{,}000`, giving :math:`\theta \approx 5 \times 10^{-4}`
per bp. Since branch lengths are :math:`O(1)` in coalescent units,
:math:`\theta\ell/2 \approx 2.5 \times 10^{-4}`, so the linear approximation
is excellent for per-site calculations.

.. code-block:: python

   def add_mutations(events, n, theta, seq_length=1):
       """Add mutations to a coalescent tree under the infinite sites model.

       For each branch in the tree, we draw a Poisson number of mutations
       (proportional to the branch length and the mutation rate), and place
       each mutation at a random position along the genome.

       Parameters
       ----------
       events : list of (time, child1, child2, parent)
           Coalescence events from simulate_coalescent().
       n : int
           Number of samples (leaf nodes, labeled 0 to n-1).
       theta : float
           Population-scaled mutation rate (4*Ne*mu).
       seq_length : int
           Sequence length in base pairs.

       Returns
       -------
       mutations : list of (position, branch_node, time)
           Each mutation has a genomic position, the node whose branch it
           sits on, and the time at which it occurred.
       """
       # Build a dictionary mapping each node to its time
       node_times = {i: 0.0 for i in range(n)}  # samples are at time 0
       children = {}
       for t, c1, c2, p in events:
           node_times[p] = t
           children[p] = (c1, c2)

       mutations = []
       root = events[-1][3]  # the root is the parent in the last event

       # For each branch (child -> parent), the length is parent_time - child_time
       for node in node_times:
           if node == root:
               continue  # the root has no parent branch
           # Find this node's parent
           for t, c1, c2, p in events:
               if c1 == node or c2 == node:
                   branch_length = node_times[p] - node_times[node]
                   # Expected mutations on this branch:
                   # theta/2 * branch_length * seq_length
                   n_muts = np.random.poisson(theta / 2 * branch_length * seq_length)
                   for _ in range(n_muts):
                       pos = np.random.uniform(0, seq_length)
                       mut_time = np.random.uniform(node_times[node], node_times[p])
                       mutations.append((pos, node, mut_time))
                   break  # found the parent, move to next node

       return sorted(mutations)

   np.random.seed(42)
   events = simulate_coalescent(10)
   mutations = add_mutations(events, 10, theta=100, seq_length=1000)
   print(f"Number of segregating sites: {len(mutations)}")

Summary
=======

You now have the core machinery of coalescent theory -- the escapement of every
Timepiece in this book. Let's take stock of what you've built:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Concept
     - Key Formula
   * - Pairwise coalescence time
     - :math:`T_2 \sim \text{Exp}(1)` (in :math:`2N` gen. units)
   * - Coalescence rate with :math:`k` lineages
     - :math:`\lambda_k = \binom{k}{2}`
   * - Expected TMRCA
     - :math:`\mathbb{E}[T_{\text{MRCA}}] = 2(1 - 1/n)`
   * - Expected lineages at time :math:`t`
     - :math:`\lambda(t) \approx \frac{n}{n + (1-n)e^{-t/2}}`
   * - Mutation probability on branch of length :math:`\ell`
     - :math:`P(\text{mut}) = 1 - e^{-\theta\ell/2}`

These results are the foundation -- the ticking mechanism at the heart of the watch.
Everything that follows, from ARGs to HMMs to full ARG inference algorithms, builds
on this fundamental clockwork.

Next: :ref:`args` -- what happens when recombination breaks the tree into many trees,
and how we represent that rich history.
