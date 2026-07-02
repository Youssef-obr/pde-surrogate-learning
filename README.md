# Classical ML Surrogates for the 1D Burgers Equation

This project studies how classical machine learning models can learn the time evolution of a physical system governed by a partial differential equation.

The equation considered here is the 1D viscous Burgers equation:

[
\partial_t u + u\partial_x u = \nu \partial_{xx}u
]

where (u(x,t)) is the state of the system and (\nu) is the viscosity.

The long-term goal is to generate reference trajectories with a numerical PDE solver, then train machine learning models to approximate the one-step evolution:

[
(u^n, \nu) \mapsto u^{n+1}
]

where (u^n) is the discretized solution at time step (n).

## Current stage

For now, the project focuses on building a reference numerical solver for the 1D Burgers equation.

The solver uses:

* Rusanov flux for the nonlinear transport term,
* central finite differences for the diffusion term,
* RK4 time integration.

This solver will later be used to generate the training data for the machine learning part.

## Preliminary solver results

The following animations show the numerical evolution of the Burgers equation for two different initial conditions.

The left animation uses a periodic sinusoidal initial condition:

[
u(x,0)=\sin(2\pi x)
]

The right animation uses a localized Gaussian initial condition:

[
u(x,0)=\exp(-x^2/2)
]

<table>
  <tr>
    <td align="center"><b>Sinusoidal initial condition</b></td>
    <td align="center"><b>Gaussian initial condition</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/burgers1Dsin.gif" width="420"/>
    </td>
    <td align="center">
      <img src="figures/burgers1Dexp.gif" width="420"/>
    </td>
  </tr>
</table>

These simulations are used to check that the solver behaves coherently before moving to the machine learning stage.

## Next direction

The next step is to focus on periodic boundary conditions and periodic initial conditions.

This choice is cleaner for the machine learning part because it avoids boundary artifacts and lets us study the main question:

> Can classical machine learning models learn a stable time evolution operator from numerical PDE trajectories?

After generating several periodic trajectories for different initial conditions and viscosity values, the project will move to classical ML baselines such as persistence, Ridge regression, PCA + Ridge, and kernel methods.
