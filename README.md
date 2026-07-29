# Classical ML Surrogates for the 1D Burgers Equation

This project explores whether classical machine learning models can learn the time evolution of a physical system governed by a nonlinear partial differential equation.

The test case is the 1D viscous Burgers equation:

$$
\partial_t u + u \partial_x u = \nu \partial_{xx}u
$$

where $u(x,t)$ is the state of the system and $\nu$ is the viscosity.

The goal is to generate reference trajectories with a numerical PDE solver, then train classical machine learning models to approximate the discrete evolution operator:

$$
(u^n, \nu) \mapsto u^{n+k}
$$

where $u^n$ is the discretized solution at time step $n$, and $k$ is the prediction horizon.

---

## Project summary

This project contains two parts:

1. A numerical solver for the 1D viscous Burgers equation.
2. A classical machine learning pipeline to learn short-time PDE evolution from generated trajectories.

The final model used in this version of the project is:

$$
\text{Kernel Ridge Regression} + \text{HGB residual correction} + \text{amplitude calibration}
$$

The model captures the global evolution of the solution and correctly follows the spatial displacement of peaks. Its main remaining limitation is that peak amplitudes are still slightly underestimated during rollout.

This version of the project stops here intentionally. The next step would be to test more advanced surrogate models, for example PCA/POD-based models, spectral features, or neural surrogates.

---

## Numerical solver

Reference trajectories are generated using a finite-volume / finite-difference solver.

The solver uses:

- Rusanov flux for the nonlinear transport term
- central finite differences for the diffusion term
- RK4 time integration
- periodic boundary conditions for the machine learning experiments

The solver is used as the ground-truth generator for supervised learning.

---

## Solver validation

Two simple initial conditions are used to visually validate the solver.

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

For the viscous Burgers equation, viscosity dissipates energy over time. As a sanity check, the discrete energy is computed as:

$$
E^n = \frac{1}{2}\sum_i (u_i^n)^2 \Delta x
$$

<p align="center">
  <img src="figures/energysin.png" width="520"/>
</p>

The observed decay confirms that the solver captures the expected dissipative behavior.

---

## Machine learning formulation

The machine learning task is supervised regression.

Each input sample is:

$$
X = (u_0^n, u_1^n, \ldots, u_{N-1}^n, \nu)
$$

and the target is:

$$
Y = (u_0^{n+k}, u_1^{n+k}, \ldots, u_{N-1}^{n+k})
$$

Random periodic initial conditions are generated as finite Fourier series:

$$
u(x,0)=\sum_{j=1}^{5} a_j \sin(2\pi j x + \theta_j)
$$

where the amplitudes $a_j$, phases $\theta_j$, and viscosity $\nu$ are sampled randomly.

The train/validation split is done by trajectory ID, so that all states from the same trajectory remain in the same split. This avoids leakage between training and validation.

---

## Models tested

The following classical models and variants were tested:

| Model / idea | Outcome |
|---|---|
| Persistence baseline | Useful sanity-check baseline, but poor rollout accuracy |
| Linear Ridge | Strong simple baseline, better than persistence |
| Polynomial Ridge | Did not improve; unstable / worse in this setup |
| Kernel Ridge Regression | Best standalone classical model |
| HGB Regressor | Did not outperform Kernel Ridge as a direct predictor |
| Residual learning | Improved peak shape visually in some cases, but worsened rollout error |
| PDE-inspired features | Improved one-step Linear Ridge, but did not improve rollout |
| HGB residual correction | Small improvement over direct Kernel Ridge |
| Amplitude calibration | Best final improvement for rollout accuracy |

The final retained model is:

$$
\hat{u}^{n+k}_{\text{final}}
=
\bar{u}
+
c
\left(
\hat{u}^{n+k}_{\text{KRR+HGB}}
-
\bar{u}
\right)
$$

where $\bar{u}$ is the spatial mean of the current state and $c$ is a scalar amplitude calibration factor fitted on validation data.

---

## Evaluation

Models are evaluated using relative L2 error:

$$
\frac{\lVert y_{\text{true}} - y_{\text{pred}} \rVert_2}{\lVert y_{\text{true}} \rVert_2}
$$

Two evaluation modes are used:

1. **One-step prediction error**: predicts one future state from the current state.
2. **Rollout error**: recursively applies the model multiple times to simulate a trajectory.

Rollout error is the most important metric because it measures whether the learned model remains stable when its own predictions are fed back as inputs.

---

## Main results

The main experiment uses:

- grid size: $N=128$
- prediction horizon: `store_every = 250`
- periodic random Fourier initial conditions
- validation split by trajectory

Representative validation results:

| Model | One-step relative L2 | Rollout relative L2 |
|---|---:|---:|
| Persistence | 0.506 | 7.55 |
| Linear Ridge | 0.277 | 0.707 |
| Kernel Ridge | 0.248 | 0.650 |
| Kernel Ridge + HGB correction | — | 0.610 |
| Kernel Ridge + HGB correction + amplitude calibration | — | **0.490** |

The final model significantly improves rollout accuracy compared to the direct Kernel Ridge model.

---

## Final rollout

The animation below shows the final model rollout on an unseen validation trajectory.

<p align="center">
  <img src="figures/rollout_krr_hgb_amplitude_store250.gif" width="620"/>
</p>

The model captures:

- the global shape of the solution,
- the direction of evolution,
- the spatial position of the main peaks,
- the dissipative smoothing behavior.

The main remaining limitation is amplitude underestimation: the predicted peaks are still slightly too conservative.

---

## Interpretation

The project shows that classical machine learning can learn a meaningful approximation of the Burgers time-evolution operator.

Kernel Ridge Regression gives the strongest standalone classical model. Adding a residual correction model and amplitude calibration improves rollout performance, but does not completely remove the peak-amplitude bias.

This behavior is expected: the models are trained with one-step supervised regression losses, which tend to produce conservative predictions. During recursive rollout, this conservativeness accumulates and leads to smoothed peaks.

The final result is therefore a useful classical ML surrogate baseline, but not a full replacement for a numerical solver.

---

## Limitations

The main limitations of the current approach are:

- the model is trained on one-step prediction, not directly on rollout loss;
- peak amplitudes are underestimated;
- recursive rollout accumulates prediction errors;
- raw grid-based regression may not be the best representation for PDE dynamics;
- the model does not explicitly enforce physical invariants beyond what it learns from data.

---

## Possible next steps

This version of the project stops at the classical surrogate baseline.

Natural next directions include:

- PCA/POD + regression on reduced coefficients;
- Fourier or spectral feature representations;
- rollout-aware training objectives;
- peak-weighted losses;
- neural operator or deep learning surrogates;
- comparison against simple numerical time-stepping baselines.

---

## Status

This project is currently paused after completing the first classical ML surrogate pipeline.

The final retained model is:

$$
\text{Kernel Ridge Regression}
+
\text{HGB residual correction}
+
\text{amplitude calibration}
$$

It provides a clean baseline for future experiments with more advanced surrogate models.
