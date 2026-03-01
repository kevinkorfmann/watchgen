.. _discoal_allele_trajectory:

=======================================
The Allele Frequency Trajectory
=======================================

   *The mainspring of the mechanism: how the beneficial allele rises through the population.*

Before we can simulate a genealogy under selection, we need to know how the
population was *structured* at every moment during the sweep. That structure is
fully determined by a single function: the allele frequency trajectory
:math:`x(t)`, which records the frequency of the beneficial allele at each point
in time.

This chapter builds the trajectory from scratch. We start with the simplest model
-- the deterministic logistic trajectory -- and then add stochasticity by
conditioning the Wright-Fisher diffusion on fixation. By the end, you will have
Python code that generates both types of trajectories and understands exactly when
each is appropriate.

Think of this chapter as winding the mainspring: once we have the trajectory,
everything else -- the structured coalescent, the sweep signatures, the
comparison with msprime -- follows mechanically.


.. note::

   **Prerequisites.** This chapter assumes familiarity with:

   - **The Wright-Fisher model** -- discrete-generation random drift with finite
     population size
   - **Differential equations** -- the logistic equation and its solution
   - **The diffusion approximation** -- modeling allele frequency changes as a
     continuous stochastic process


Step 1: The Deterministic Trajectory
=======================================

Start with the simplest possible model. Suppose a beneficial allele with selection
coefficient :math:`s` (genic/additive) is present at frequency :math:`x` in a
large population. In the next generation, the expected change in frequency is:

.. math::

   \Delta x = \frac{s \, x(1 - x)}{1 + s \, x} \approx s \, x(1 - x)

The approximation holds for small :math:`s` (which is almost always true in
population genetics: :math:`s \sim 0.01` is already considered strong). This gives
us the **logistic differential equation**:

.. math::

   \frac{dx}{dt} = s \, x(1 - x)

This is the standard logistic growth equation, with :math:`s` playing the role of
the growth rate. Its solution is:

.. math::

   x(t) = \frac{1}{1 + \frac{1 - x_0}{x_0} \, e^{-st}}

where :math:`x_0 = x(0)` is the initial frequency. For a hard sweep,
:math:`x_0 = 1/(2N)`.

.. admonition:: Calculus Aside -- Solving the logistic equation

   Separate variables: :math:`\frac{dx}{x(1-x)} = s \, dt`. Use partial
   fractions: :math:`\frac{1}{x(1-x)} = \frac{1}{x} + \frac{1}{1-x}`.
   Integrate both sides:
   :math:`\ln\frac{x}{1-x} = st + C`. Exponentiate and solve for :math:`x`:

   .. math::

      x(t) = \frac{x_0 \, e^{st}}{1 - x_0 + x_0 \, e^{st}}
            = \frac{1}{1 + \frac{1 - x_0}{x_0} \, e^{-st}}

   This is the classic **sigmoid curve**: slow at the boundaries (where either
   :math:`x` or :math:`1-x` is small), fastest at :math:`x = 0.5` (where the
   "logistic force" :math:`x(1-x)` is maximal).

**Sweep duration.** How long does the sweep take? The time from frequency
:math:`\epsilon` to frequency :math:`1 - \epsilon` is:

.. math::

   T_{\text{sweep}} = \frac{1}{s} \ln\frac{(1-\epsilon)^2}{\epsilon^2}
                     \approx \frac{2}{s} \ln\frac{1}{\epsilon}

For a hard sweep from :math:`1/(2N)` to :math:`1 - 1/(2N)`:

.. math::

   T_{\text{sweep}} \approx \frac{2 \ln(2N)}{s}

With :math:`N = 10{,}000` and :math:`s = 0.01`, the sweep takes about
:math:`2 \times 9.9 / 0.01 \approx 2{,}000` generations. In coalescent units
(:math:`2N = 20{,}000` generations), that is :math:`0.1` coalescent time units --
a brief but intense event.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   def deterministic_trajectory(s, N, x0=None, dt=1.0):
       """Generate a deterministic (logistic) allele frequency trajectory.

       Parameters
       ----------
       s : float
           Selection coefficient (genic/additive).
       N : int
           Diploid effective population size.
       x0 : float, optional
           Initial frequency. Defaults to 1/(2N) for a hard sweep.
       dt : float
           Time step in generations.

       Returns
       -------
       trajectory : ndarray, shape (T,)
           Allele frequency at each generation, from origin to fixation.
       """
       if x0 is None:
           x0 = 1.0 / (2 * N)

       trajectory = [x0]
       x = x0
       t = 0
       while x < 1.0 - 1.0 / (2 * N):
           t += dt
           x = 1.0 / (1.0 + ((1.0 - x0) / x0) * np.exp(-s * t))
           trajectory.append(x)
       trajectory.append(1.0)  # fixation
       return np.array(trajectory)

   # Example: strong sweep (2Ns = 1000, so s = 0.05 for N=10000)
   N = 10_000
   s = 0.05
   traj = deterministic_trajectory(s, N)
   print(f"Sweep duration: {len(traj)} generations")
   print(f"In coalescent units (2N gen): {len(traj) / (2*N):.4f}")
   print(f"Theory: 2*ln(2N)/s = {2*np.log(2*N)/s:.0f} generations")

The deterministic trajectory is a clean sigmoid. But nature is not deterministic:
allele frequencies **fluctuate** due to finite population size (genetic drift),
especially at the boundaries where the allele is very rare or very common. To
capture this, we need the stochastic trajectory.


Step 2: The Wright-Fisher Diffusion with Selection
====================================================

In a finite population of :math:`2N` haploid individuals, the number of copies of
the beneficial allele follows a **Wright-Fisher process** with selection. If the
current number of copies is :math:`k`, the expected number in the next generation
is:

.. math::

   \mathbb{E}[k'] = 2N \cdot \frac{k(1 + s)}{k(1 + s) + (2N - k)}
                   \approx 2N \cdot \frac{k(1 + s)}{2N + ks}

and the actual number :math:`k'` is drawn from a Binomial with this expected
frequency.

In the diffusion limit (large :math:`N`, small :math:`s`, :math:`Ns` finite),
the allele frequency :math:`x = k/(2N)` follows the **Wright-Fisher diffusion**:

.. math::

   dx = s \, x(1 - x) \, dt + \sqrt{\frac{x(1-x)}{2N}} \, dW

where :math:`dW` is a Wiener process increment. The first term is the
**deterministic force** (selection pushes :math:`x` upward), and the second is
the **stochastic force** (drift creates random fluctuations proportional to
:math:`\sqrt{x(1-x)/(2N)}`).

.. admonition:: Probability Aside -- The diffusion approximation

   The diffusion approximation replaces the discrete Wright-Fisher model (which
   tracks integer counts :math:`k \in \{0, 1, \ldots, 2N\}`) with a continuous
   process (:math:`x \in [0, 1]`). This is valid when :math:`N` is large and
   changes per generation are small.

   The key quantities are the **infinitesimal mean** (drift coefficient):
   :math:`\mu(x) = s \, x(1 - x)`, and the **infinitesimal variance** (diffusion
   coefficient): :math:`\sigma^2(x) = x(1 - x) / (2N)`. These fully
   characterize the process.


Step 3: Conditioning on Fixation
==================================

There is a problem: most beneficial mutations are **lost to drift** before they
can establish. The fixation probability of a new mutation with selection
coefficient :math:`s` is:

.. math::

   h(x) = \frac{1 - e^{-2Ns \cdot x}}{1 - e^{-2Ns}}

For a new mutation (:math:`x = 1/(2N)`):

.. math::

   h\!\left(\frac{1}{2N}\right) \approx \frac{s}{1 - e^{-2Ns}}
   \approx s \quad \text{for } 2Ns \gg 1

So even a strongly beneficial mutation (:math:`s = 0.01`) fixes only about 1% of
the time. But we are simulating **selective sweeps** -- events where the allele
*did* fix. We need to simulate the allele frequency trajectory **conditioned on
fixation**.

This conditioning changes the dynamics. Intuitively, the conditioned trajectory is
biased *away* from 0: trajectories that would have been absorbed at 0 (lost) are
excluded, so the surviving trajectories have an extra upward push. This extra push
is called **fictitious selection** (Zhao, Charlesworth & Robin, 2013).

The conditioned diffusion has modified drift:

.. math::

   dx = \left[s \, x(1 - x) + \frac{x(1 - x)}{2N} \cdot \frac{h'(x)}{h(x)}\right] dt
        + \sqrt{\frac{x(1-x)}{2N}} \, dW

The extra term :math:`\frac{x(1-x)}{2N} \cdot \frac{h'(x)}{h(x)}` is the
fictitious selection. Let us compute it.

The derivative of the fixation probability is:

.. math::

   h'(x) = \frac{2Ns \cdot e^{-2Nsx}}{1 - e^{-2Ns}}

So:

.. math::

   \frac{h'(x)}{h(x)} = \frac{2Ns \cdot e^{-2Nsx}}{1 - e^{-2Nsx}}

This ratio is large when :math:`x` is small (the conditioning strongly pushes the
allele away from loss) and small when :math:`x` is large (near fixation, the
conditioning adds little extra push because the allele is already likely to fix).

.. admonition:: Closing a confusion gap -- Why "fictitious" selection?

   The fictitious selection is not a real biological force. It is a mathematical
   consequence of conditioning on the outcome (fixation). Think of it this way:
   if you only observe coin-flip sequences that end with "heads," those sequences
   will appear to be biased toward heads, even if the coin is fair. The bias is
   real in the conditioned ensemble but fictitious in the underlying process.

   In our case, the "coin" is the Wright-Fisher process, and "ending with heads"
   is fixation. The conditioned trajectories look as if there were an extra
   selective force pushing the allele upward.


Step 4: The Conditioned Jump Process
======================================

discoal does not directly simulate the conditioned diffusion. Instead, it uses a
**discrete jump process** that approximates the conditioned diffusion in each
generation. This is the approach of Coop & Griffiths (2004) and Eriksson,
Fernstrom & Mehlig (2008).

The idea: at each generation, the allele count :math:`k` (out of :math:`2N`)
either increases by 1 or decreases by 1. The probability of increasing,
conditioned on eventual fixation, is:

.. math::

   p_+(k) = \frac{k(2N - k) \cdot h(k+1)}{(k+1)(2N-k-1) \cdot h(k) + k(2N-k) \cdot h(k+1)}

where :math:`h(k) = h(k/(2N))` is the fixation probability from count :math:`k`.

This jump process is simulated **backward** from fixation (:math:`k = 2N`):
starting at :math:`k = 2N`, at each step we decrease or increase the count with
probabilities that ensure the trajectory, read forward, is a valid conditioned
Wright-Fisher sample.

.. admonition:: Probability Aside -- Why simulate backward?

   Simulating backward from fixation is natural because we *know* the endpoint
   (fixation). The conditioned probability of going from :math:`k` to
   :math:`k-1` backward is equivalent to the conditioned probability of going
   from :math:`k-1` to :math:`k` forward, adjusted by the fixation probabilities.
   This avoids the rejection problem of forward simulation (where most
   trajectories would be lost and rejected).

.. code-block:: python

   def fixation_probability(x, two_N_s):
       """Fixation probability from frequency x under genic selection.

       Parameters
       ----------
       x : float
           Current allele frequency.
       two_N_s : float
           Scaled selection coefficient 2Ns.

       Returns
       -------
       h : float
           Probability of eventual fixation.
       """
       if abs(two_N_s) < 1e-10:
           return x  # neutral case: fixation prob = current frequency
       return (1.0 - np.exp(-two_N_s * x)) / (1.0 - np.exp(-two_N_s))

   def stochastic_trajectory(s, N, x0=None, rng=None):
       """Generate a stochastic allele frequency trajectory conditioned on fixation.

       Uses the conditioned jump process approximation (Coop & Griffiths 2004).
       Simulates backward from fixation, then reverses.

       Parameters
       ----------
       s : float
           Selection coefficient (genic).
       N : int
           Diploid effective population size.
       x0 : float, optional
           Initial frequency. Defaults to 1/(2N) for a hard sweep.
       rng : np.random.Generator, optional
           Random number generator.

       Returns
       -------
       trajectory : ndarray
           Allele frequencies from x0 to 1.0, one entry per generation.
       """
       if rng is None:
           rng = np.random.default_rng(42)
       if x0 is None:
           x0 = 1.0 / (2 * N)

       two_N = 2 * N
       two_N_s = two_N * s
       k0 = max(1, int(x0 * two_N))

       # Precompute fixation probabilities for all possible allele counts
       h = np.array([fixation_probability(k / two_N, two_N_s) for k in range(two_N + 1)])

       # Simulate backward from fixation (k = 2N)
       k = two_N
       counts = [k]

       # First step back from fixation: must decrease by 1
       k -= 1
       counts.append(k)

       while k > k0:
           if k <= 0:
               break

           # Going backward from count k, the probability that the
           # previous count was k-1 (meaning it went up to k going forward):
           p_was_lower = k * (two_N - k) * h[min(k + 1, two_N)]
           p_was_higher = (k + 1) * (two_N - k - 1) * h[k] if k < two_N - 1 else 0

           denom = p_was_lower + p_was_higher
           if denom == 0:
               break
           p_down = p_was_lower / denom

           if rng.random() < p_down:
               k -= 1
           else:
               k += 1
               # Prevent going above 2N
               k = min(k, two_N - 1)
           counts.append(k)

       # Reverse to get forward-time trajectory
       counts.reverse()
       trajectory = np.array(counts) / two_N
       return trajectory

   # Example: compare deterministic and stochastic trajectories
   N = 10_000
   s = 0.01  # 2Ns = 200

   det_traj = deterministic_trajectory(s, N)
   stoch_traj = stochastic_trajectory(s, N)

   print(f"Deterministic sweep: {len(det_traj)} generations")
   print(f"Stochastic sweep:    {len(stoch_traj)} generations")


Step 5: Comparing the Two Trajectory Modes
=============================================

When should you use each trajectory mode?

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - Deterministic
     - Stochastic
   * - Speed
     - Very fast (closed-form)
     - Slower (one random draw per generation)
   * - Boundary behavior
     - Unrealistic (smooth approach to 0 and 1)
     - Realistic (captures stochastic loss/fixation dynamics)
   * - When to use
     - :math:`2Ns \gg 1` (strong selection), or for quick approximation
     - :math:`2Ns` moderate, or when drift near boundaries matters
   * - Typical :math:`2Ns`
     - :math:`> 500`
     - :math:`10 - 500`

The key difference is at the **boundaries**. The deterministic trajectory smoothly
approaches 0 and 1, spending many generations at very low and very high
frequencies. In reality, the allele jumps quickly through the low-frequency phase
(either lost or escaping to higher frequency by chance) and the high-frequency
phase (rapidly fixing by a mix of selection and drift). The stochastic trajectory
captures this.

.. code-block:: python

   def compare_trajectories(s, N, n_stochastic=5, seed=42):
       """Plot deterministic vs stochastic trajectories side by side.

       Parameters
       ----------
       s : float
           Selection coefficient.
       N : int
           Diploid effective population size.
       n_stochastic : int
           Number of stochastic trajectories to overlay.
       seed : int
           Random seed.
       """
       rng = np.random.default_rng(seed)

       det = deterministic_trajectory(s, N)
       det_time = np.arange(len(det))

       fig, axes = plt.subplots(1, 2, figsize=(12, 5))

       # Left panel: deterministic
       axes[0].plot(det_time, det, 'k-', linewidth=2)
       axes[0].set_xlabel('Generation')
       axes[0].set_ylabel('Beneficial allele frequency x(t)')
       axes[0].set_title(f'Deterministic (2Ns = {2*N*s:.0f})')
       axes[0].set_ylim(-0.05, 1.05)
       axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.3)

       # Right panel: stochastic
       for i in range(n_stochastic):
           stoch = stochastic_trajectory(s, N, rng=rng)
           stoch_time = np.arange(len(stoch))
           axes[1].plot(stoch_time, stoch, alpha=0.6, linewidth=1)
       axes[1].plot(det_time, det, 'k--', linewidth=1, label='deterministic')
       axes[1].set_xlabel('Generation')
       axes[1].set_title(f'Stochastic (2Ns = {2*N*s:.0f})')
       axes[1].set_ylim(-0.05, 1.05)
       axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.3)
       axes[1].legend()

       plt.tight_layout()
       return fig

   # Strong selection: stochastic tracks deterministic closely
   # compare_trajectories(0.05, 10_000)

   # Moderate selection: stochastic deviates noticeably
   # compare_trajectories(0.005, 10_000)


Step 6: Trajectory Duration and Its Effect on the Genealogy
=============================================================

The sweep duration matters because it determines **how long** the structured
coalescent operates. A longer sweep gives more opportunity for:

- **Coalescence within the** :math:`B` **bottleneck** -- more time at low
  :math:`x(t)` means more complete destruction of diversity.
- **Recombination escape** -- more time for lineages at linked loci to recombine
  off the beneficial background.

The sweep duration in generations is approximately:

.. math::

   T_{\text{sweep}} \approx \frac{2 \ln(2N)}{s} = \frac{2 \ln(2N) \cdot 2N}{\alpha}

where :math:`\alpha = 2Ns`. In coalescent time units (:math:`2N` generations):

.. math::

   T_{\text{sweep}}^{\text{coal}} \approx \frac{2 \ln(2N)}{\alpha}

For :math:`N = 10{,}000` and :math:`\alpha = 1000` (:math:`s = 0.05`):

.. math::

   T_{\text{sweep}}^{\text{coal}} \approx \frac{2 \times 9.9}{1000} \approx 0.02

That is 2% of one coalescent time unit -- a flash. The sweep is brief but
catastrophic to diversity near the selected site.

.. code-block:: python

   def sweep_duration_table(N, alphas):
       """Print a table of sweep durations for different selection strengths.

       Parameters
       ----------
       N : int
           Diploid effective population size.
       alphas : list of float
           Values of 2Ns to tabulate.
       """
       two_N = 2 * N
       print(f"{'alpha (2Ns)':>12} {'s':>10} {'T_sweep (gen)':>15} "
             f"{'T_sweep (coal)':>16} {'T_sweep (4N gen)':>18}")
       print("-" * 75)
       for alpha in alphas:
           s = alpha / two_N
           T_gen = 2 * np.log(two_N) / s
           T_coal = T_gen / two_N
           T_4N = T_gen / (4 * N)
           print(f"{alpha:12.0f} {s:10.5f} {T_gen:15.0f} "
                 f"{T_coal:16.4f} {T_4N:18.6f}")

   N = 10_000
   alphas = [10, 50, 100, 500, 1000, 5000]
   sweep_duration_table(N, alphas)

   # Output:
   # alpha (2Ns)          s   T_sweep (gen)   T_sweep (coal)   T_sweep (4N gen)
   # ---------------------------------------------------------------------------
   #           10    0.00050           39588           1.9794           0.989700
   #           50    0.00250            7918           0.3959           0.197940
   #          100    0.00500            3959           0.1979           0.098970
   #          500    0.02500             792           0.0396           0.019794
   #         1000    0.05000             396           0.0198           0.009897
   #         5000    0.25000              79           0.0040           0.001979

The pattern is clear: stronger selection means a shorter, sharper sweep. The
genealogical effect is concentrated in a brief window of time, but during that
window, coalescence rates within the :math:`B` background are enormous.


What Comes Next
=================

With the trajectory :math:`x(t)` in hand -- whether deterministic or stochastic --
we are ready to build the structured coalescent that runs on top of it. The
trajectory is the schedule; the structured coalescent is the machine that follows
the schedule.

In the next chapter, we wire the trajectory into the coalescent by partitioning
lineages into two backgrounds and computing rates that depend on :math:`x(t)` at
every time step.
