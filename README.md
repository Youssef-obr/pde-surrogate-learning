## Amplitude calibration

Kernel Ridge Regression with HGB residual correction captured the global shape of the solution and the spatial position of the peaks, but it still underestimated peak amplitudes.

Let the corrected prediction before amplitude calibration be:

```math
\tilde{u}^{n+k}
=
\hat{u}_{\mathrm{KRR}}^{n+k}
+
\hat{e}_{\mathrm{HGB}}^{n+k}
```

where $\hat{e}_{\mathrm{HGB}}^{n+k}$ is the residual predicted by the HGB correction model.

The final prediction is obtained by scaling the deviation from the spatial mean:

```math
\hat{u}_{\mathrm{final}}^{n+k}
=
\bar{u}^n
+
c
\left(
\tilde{u}^{n+k}
-
\bar{u}^n
\right)
```

with

```math
\bar{u}^n
=
\frac{1}{N}
\sum_{i=0}^{N-1} u_i^n
```

The scalar $c$ is fitted on validation predictions by least squares:

```math
c
=
\frac{
\sum_s
\left\langle
\tilde{u}_s^{n+k} - \bar{u}_s^n,
u_{s,\mathrm{true}}^{n+k} - \bar{u}_s^n
\right\rangle
}{
\sum_s
\left\lVert
\tilde{u}_s^{n+k} - \bar{u}_s^n
\right\rVert_2^2
}
```

This correction was added because the model was already predicting the correct global shape and peak locations, but its amplitudes were too conservative. Scaling around the spatial mean boosts peaks and valleys without shifting the whole solution vertically.
