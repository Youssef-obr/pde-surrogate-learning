# Classical ML Surrogates for the 1D Burgers Equation

This repository studies whether lightweight classical machine learning models can approximate the time evolution of a nonlinear PDE when trained directly on raw spatial grid states.

The test case is the one-dimensional viscous Burgers equation:

```math
\partial_t u + u\partial_x u = \nu \partial_{xx}u
```

where \(u(x,t)\) is the solution field and \(\nu\) is the viscosity.

The learning objective is to approximate a discrete flow map:

```math
(u^n, \nu) \mapsto u^{n+k}
```

where \(u^n \in \mathbb{R}^N\) is the discretized solution at time step \(n\), and \(k\) is the prediction horizon.

This repository is Part I of a larger project. The goal is to build a clean raw-grid classical ML baseline, evaluate its rollout behavior, and identify the limitations that motivate a more systematic reduced-coordinate study in Part II.

---

## Numerical solver

Reference trajectories are generated using:

- Rusanov flux for the nonlinear convection term;
- central finite differences for the diffusion term;
- RK4 time integration;
- periodic boundary conditions.

The solver is first checked on simple initial conditions.

<table>
  <tr>
    <td align="center"><b>Sinusoidal initial condition</b></td>
    <td align="center"><b>Localized Gaussian initial condition</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/burgers1Dsin.gif" width="360"/>
    </td>
    <td align="center">
      <img src="figures/burgers1Dexp.gif" width="360"/>
    </td>
  </tr>
</table>

For the viscous Burgers equation, the discrete energy is expected to decay over time:

```math
E^n = \frac{1}{2}\sum_i (u_i^n)^2\Delta x
```

<p align="center">
  <img src="figures/energysin.png" width="520"/>
</p>

This decay is used as a sanity check that the numerical solver captures the dissipative behavior of viscous Burgers dynamics.

---

## Dataset and learning task

Random periodic initial conditions are generated as finite Fourier sums:

```math
u(x,0) = \sum_{j=1}^{5} a_j\sin(2\pi jx + \theta_j)
```

where the amplitudes \(a_j\), phases \(\theta_j\), and viscosity values \(\nu\) are sampled randomly.

Each supervised input is:

```math
X = (u_0^n, u_1^n, \ldots, u_{N-1}^n, \nu)
```

and the target is:

```math
Y = (u_0^{n+k}, u_1^{n+k}, \ldots, u_{N-1}^{n+k})
```

The split is performed by `trajectory_id`, so that all states from the same trajectory remain in the same split. This avoids leakage between training and validation.

Main experimental parameters:

| Parameter | Value |
|---|---:|
| Grid size | \(N = 128\) |
| Time step | \(\Delta t = 10^{-4}\) |
| Solver steps | `5000` |
| Final time | \(T = 0.5\) |
| Prediction horizon | `store_every = 250` |
| Physical prediction time | \(250\Delta t = 0.025\) |
| Number of trajectories | `30` |
| Train / validation trajectories | `25 / 5` |
| Viscosity values | `[0.01, 0.02, 0.05, 0.1]` |
| Random seed | `42` |

---

## Models

The tested classical models are:

- Persistence baseline;
- Linear Ridge regression;
- Polynomial Ridge regression;
- Kernel Ridge regression with RBF kernel;
- Kernel Ridge with HGB residual correction;
- Kernel Ridge with HGB correction and rollout-calibrated amplitude gain.

Kernel Ridge Regression is used as the main baseline because it gives the best standalone one-step validation error among the tested classical models:

```math
\hat{u}_{\mathrm{KRR}}^{n+k}
=
f_{\mathrm{KRR}}(u^n,\nu)
```

---

## Residual correction

After fitting Kernel Ridge, a second model is trained to correct its residual error.

The residual target is:

```math
e^{n+k}
=
u_{\mathrm{true}}^{n+k}
-
\hat{u}_{\mathrm{KRR}}^{n+k}
```

The correction model learns:

```math
\hat{e}_{\mathrm{HGB}}^{n+k}
=
f_{\mathrm{HGB}}(u^n,\nu,\hat{u}_{\mathrm{KRR}}^{n+k})
```

The corrected prediction is:

```math
\tilde{u}^{n+k}
=
\hat{u}_{\mathrm{KRR}}^{n+k}
+
\hat{e}_{\mathrm{HGB}}^{n+k}
```

The HGB correction is used as a nonlinear residual model. The motivation is that, after Kernel Ridge prediction, the remaining error may still depend nonlinearly on the current state, the viscosity, and the predicted future state. HistGradientBoosting provides a classical non-neural way to test whether such residual structure can be captured.

In this experiment, HGB residual correction does not improve one-step error by itself:

| Model | One-step MSE | One-step relative \(L^2\) |
|---|---:|---:|
| Kernel Ridge | **0.003978** | **0.248** |
| Kernel Ridge + HGB correction | 0.004418 | 0.261 |

It is therefore interpreted as a residual-correction experiment rather than a clear standalone improvement.

---

## Rollout-calibrated amplitude gain

During rollout, the model is recursively applied. Small amplitude errors can accumulate because the model feeds its own predictions back as future inputs.

After residual correction, the final rollout model applies a scalar gain to the deviation from the initial spatial mean:

```math
\hat{u}_{\mathrm{final}}^{n+k}
=
\bar{u}^0
+
g
\left(
\tilde{u}^{n+k}
-
\bar{u}^0
\right)
```

where

```math
\bar{u}^0
=
\frac{1}{N}
\sum_{i=0}^{N-1} u_i^0
```

The gain is selected by validation rollout error minimization:

```math
g^\star
=
\arg\min_g
\frac{
\left\lVert
u_{\mathrm{true}}^{\mathrm{rollout}}
-
u_{\mathrm{pred}}^{\mathrm{rollout}}(g)
\right\rVert_2
}{
\left\lVert
u_{\mathrm{true}}^{\mathrm{rollout}}
\right\rVert_2
}
```

The tested values were:

```math
g \in \{0.50, 0.55, 0.60, \ldots, 1.50\}
```

The selected rollout gain was:

```math
g^\star = 0.85
```

with mean validation rollout relative \(L^2\):

```math
0.6289823688059181
```

This gain slightly damps amplitudes. It is selected from validation rollout performance.

---

## Evaluation

The main metric is the relative \(L^2\) error:

```math
\frac{
\lVert u_{\mathrm{true}} - u_{\mathrm{pred}} \rVert_2
}{
\lVert u_{\mathrm{true}} \rVert_2
}
```

Two evaluation regimes are used:

- **one-step prediction**, where the model predicts one future state;
- **rollout prediction**, where the model is recursively applied to generate a full trajectory.

Rollout evaluation is the more important test because errors compound when the model uses its own predictions as future inputs.

---

## Results

| Model | One-step relative \(L^2\) | Rollout relative \(L^2\) |
|---|---:|---:|
| Persistence | 0.506 | 7.55 |
| Linear Ridge | 0.277 | 0.707 |
| Polynomial Ridge | 0.427 | — |
| Kernel Ridge | **0.248** | 0.650 |
| Kernel Ridge + HGB correction | 0.261 | — |
| Kernel Ridge + HGB correction + gain | — | **0.455** on selected rollout / 0.629 mean validation rollout |

Kernel Ridge is the best standalone one-step model. The corrected model gives the best selected rollout, but the mean validation rollout error remains significant.

The main conclusion is that classical ML models can learn meaningful short-horizon Burgers dynamics, but raw-grid recursive rollouts remain fragile.

---

## Final rollout

The animation below compares the corrected model rollout with the reference numerical trajectory on an unseen validation trajectory.

<p align="center">
  <img src="figures/rollout_kernel_ridge_store250.gif" width="620"/>
</p>

The model captures the qualitative form of the solution, including the global shape and peak locations, but still exhibits amplitude and long-horizon rollout errors.

---

## Diagnostic plots

Pointwise diagnostic plots are used to inspect calibration errors beyond global relative \(L^2\).

<table>
  <tr>
    <td align="center"><b>True vs predicted values</b></td>
    <td align="center"><b>Residuals vs predicted values</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/true_vs_pred_scatter.png" width="420"/>
    </td>
    <td align="center">
      <img src="figures/residuals_vs_pred.png" width="420"/>
    </td>
  </tr>
</table>

The residual is defined as:

```math
r = u_{\mathrm{true}} - u_{\mathrm{pred}}
```

These plots help reveal whether the model is globally calibrated or whether errors concentrate around large positive and negative amplitudes.

---

## Prediction cost

A practical surrogate should eventually be faster than recomputing the PDE solution. After training, prediction time was compared with recomputing the same validation trajectories numerically.

| Method | Time on validation trajectories |
|---|---:|
| ML surrogate prediction | 704.28 s |
| Numerical solver | 65.26 s |
| Speedup | 0.09x |

In this implementation, the corrected raw-grid surrogate is not faster than the numerical solver. This is mainly due to the cost of Kernel Ridge prediction, the HGB residual correction, and repeated raw-grid autoregressive rollout.

This result does not invalidate the learning experiment. It shows that this first raw-grid model is a diagnostic baseline rather than a deployable fast surrogate.

---

## Interpretation

This repository establishes a first raw-grid classical ML baseline for Burgers surrogate modeling.

The main findings are:

1. Classical ML can approximate short-horizon Burgers evolution.
2. One-step error is not sufficient to judge surrogate quality.
3. Rollout prediction exposes instability and amplitude miscalibration.
4. Raw-grid corrected models are not necessarily computationally efficient.
5. Better coordinate representations are likely needed for stable and efficient classical surrogates.

---

## Toward Part II

Part II will investigate whether these failure modes are partly caused by learning directly in raw physical grid coordinates.

The planned research question is:

```math
\text{Do reduced coordinates improve classical ML rollouts for Burgers dynamics?}
```

The next study will compare:

- raw-grid states;
- PCA / POD coefficients;
- Fourier coefficients.

The evaluation will include:

- one-step relative \(L^2\);
- rollout relative \(L^2\);
- extrema error;
- energy decay error;
- spectral error;
- rollout stability;
- prediction time.

The goal is to understand why raw-grid classical ML surrogates can predict visually plausible trajectories while still failing on physically important quantities such as amplitudes, spectra, and long-time behavior.

---

## Relation to existing work

Burgers surrogate modeling is a standard benchmark in scientific machine learning and neural operator literature. This repository does not claim to introduce a new Burgers solver or a new surrogate architecture.

Instead, it serves as a student research baseline focused on lightweight classical ML models trained directly on raw grid data. The follow-up project will connect this baseline to reduced-order and operator-learning ideas more systematically.

Relevant references for the next stage include:

- Fourier Neural Operator for Parametric Partial Differential Equations;
- PDEBench: An Extensive Benchmark for Scientific Machine Learning;
- non-intrusive reduced-order modeling and regression-based surrogate methods for PDEs.

---

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python burgers_structured/main.py
```

The program will:

1. run a solver demo;
2. generate numerical Burgers trajectories;
3. train classical ML models;
4. evaluate one-step prediction;
5. train the residual correction;
6. apply the fixed rollout gain;
7. compare prediction cost with numerical solving;
8. save the rollout GIF;
9. show diagnostic plots.

---

## Status

This repository completes Part I of the project:

```math
\text{Raw-grid classical ML baseline for Burgers surrogate modeling}
```

The conclusion is:

```math
\text{Classical ML can learn meaningful Burgers dynamics, but raw-grid corrected rollouts remain limited in accuracy and efficiency.}
```

Part II will focus on reduced-coordinate surrogate modeling and systematic failure-mode analysis.
