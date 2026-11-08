# Conditional Model Scaling and Limits of Interpretation {#scaling-laws-and-system-level-behavior}

## Descriptive Analytical Scaling

Parameter sweeps describe the selected closed-form models. They do not measure physiological scaling, solver convergence, or hardware latency. Current fit coefficients and intervals are reported only in the completed run's `Generated.md`; no fixed exponent or goodness-of-fit value is copied here.

A finite-range power-law fit $y=ax^b$ summarizes a curve over its stated range. A large $b$ does not convert a power law into an exponential, and a nearly constant baseline-inclusive response does not imply zero computational or mechanical cost. Compute-only and baseline-inclusive curves answer different questions and must be labeled separately.

## AntBody: Solver and Joint Dependence

In the selected base model, joint count affects both contact work and quadratic control work. The `pgs` iteration rule is $\min(50,\max(10,C))$; contact work is proportional to contact count times that iteration count at fixed $J$. Its piecewise operation counts are not a measured $C^{1.5}$ solver law. The other supported models are `si` and `ldlt`, not LCP/MLCP aliases. Their formula differences alone establish neither an accuracy ranking nor a preferred contact threshold.

Adding a fixed baseline can flatten energy variation relative to compute-only scaling. The manifest baseline is an assumed 0.50 W, not 50 mW. Morphological complexity is not physically free: mechanics, sensing, and hardware utilization must be considered on the same time basis. A nominal 100 Hz control rate is an input; real-time feasibility needs measured latency and deadline-miss distributions on identified hardware.

## AntBrain: Activity and Connectivity Assumptions

At fixed $N_{KC}$ and $\rho$, base channel processing grows with $K$ and channel state traffic grows with $KH$. A fixed sparse-population term can dominate those increments over a finite range, but there is no general parameter-independent neural energy law. The `random` and `structured` operation models use different assumed connectivity costs; neither is a measured ant topology. There is no supported biological-over-small-world efficiency comparison.

The example activity fraction $\rho=0.02$ is an illustrative assumption. Lower modeled activity reduces some counters, but representational capacity, information loss, learning quality, and task performance are not evaluated by these counters. A sparsity optimum requires a task and data, not energy alone.

## AntMind: Exact Trees Versus Bounded Policies

The selected base model sums tree nodes through horizon $H_p$ with branching factor $B$. For $B>1$, exact enumeration grows exponentially in horizon. Enhanced Mind is a distinct policy-count model with a uniform finite budget (default 1000 policies), or exact leaf enumeration when `max_policies=None`. A capped model can saturate its policy-count contribution; it does not exhibit the uncapped tree's growth by necessity. No universal 15-step real-time boundary follows from either model.

Inverse parameter values used in energy–proxy plots are declared mathematical proxies, not measured dexterity, accuracy, reward, or throughput. A Pareto frontier requires explicit objective directions and exclusion of dominated points; a sorted curve alone is not evidence of optimality.

## Critical Behavior: Not Established

The released analytical workflow supplies no evidence of a second-order transition at 20 contacts, a first-order transition at activity fraction 0.02, or a planning divergence at horizon 15. Earlier numerical critical exponents (0.67, 0.8, 1.8), their intervals, a tenfold iteration jump, hysteresis, phase coexistence, universality classes, and finite-size critical-point shifts are not retained as results. The selected memoryless closed-form sweeps do not test path dependence or hysteresis.

A future critical-behavior study would need an explicit dynamical model and order parameter, bidirectional sweeps for path dependence, sufficient sizes and resolution, competing noncritical explanations, and independent observations. Piecewise counter rules, fit residuals, and policy-budget saturation should first be explained as model conventions, not physical phase transitions.

## Design Interpretation

The framework can compare assumed operation/memory prices and workload dimensions while holding other inputs fixed. Such comparisons motivate device-specific profiling, sensor compression, activity-aware processing, and bounded planning experiments. Compute-aware stopping is supported as a distinct motion-planning research direction [@sudhakar2020balancing], not demonstrated behavior of this analytical sweep. No hardware-independent savings, biological superiority, or universal scaling collapse is inferred.
