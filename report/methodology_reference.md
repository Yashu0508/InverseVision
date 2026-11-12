# Methodology Reference

## Forward model and notation

Let `X` be a sharp grayscale image of shape `r x c`, `x = vec_C(X)` its
C-order vectorization, and `n = rc`. The saved observation follows

\[
b = A x + \epsilon,
\]

where `A` is a zero-boundary Gaussian blur operator and `epsilon` is additive
Gaussian noise. The normalized one-dimensional Gaussian kernel has support
`k` and standard deviation `sigma`. The edge-truncated kernel is deliberately
not renormalized.

With `B_y` of shape `r x r` and `B_x` of shape `c x c`, implementation uses

\[
Y=B_yXB_x^T, \qquad \operatorname{vec}_C(Y)=(B_y\otimes B_x)\operatorname{vec}_C(X).
\]

Therefore `A = B_y kron B_x` has shape `n x n`. The matrix-free action
reshapes a C-order vector, applies `B_y X B_x.T`, then re-flattens C-order.
Its exact adjoint is `vec_C(B_y.T Z B_x)`. It does not assume symmetry.

## Spectral analysis

For `A=U Sigma V.T`, singular values are nonincreasing. With relative
tolerance `tau`, a usable singular value exceeds `tau sigma_1`; the condition
number is `sigma_max/sigma_min` over usable positive values. Separable spectra
are all pairwise products of the two one-dimensional spectra. Picard analysis
uses `u_i.T b` and exposes the noise amplification caused by division by small
`sigma_i`.

## Reconstruction and regularization

Baselines solve the unregularized inverse or least-squares problem. QR solves
least squares; normal equations solve `A.T A x=A.T b` and may square
conditioning sensitivity.

\[
x_k=\sum_{i=1}^k\frac{u_i^Tb}{\sigma_i}v_i
\]

is TSVD with retained rank `k`. Standard L2 Tikhonov solves

\[
x_\lambda=\arg\min_x\|Ax-b\|_2^2+\lambda^2\|x\|_2^2
=\sum_i\frac{\sigma_i}{\sigma_i^2+\lambda^2}(u_i^Tb)v_i.
\]

The augmented formulation is `[A; lambda I]x=[b;0]`. TSVD uses hard spectral
retention; Tikhonov uses smooth retention `sigma_i^2/(sigma_i^2+lambda^2)`.

## Parameter selection and iteration

L-curve selection maximizes finite discrete curvature of the log residual and
log solution norms. GCV is

\[
G=\frac{\|Ax-b\|_2^2/n}{(1-d/n)^2},
\]

where `d` is effective degrees of freedom. The discrepancy target is
`eta sigma_epsilon sqrt(n)`; saved experiments use `eta=1.05`. All selectors
use only operator, observation, SVD, and candidate grid, never ground truth.

LSQR and CGLS use only matrix-free forward and adjoint actions. Damped LSQR
uses `damp=lambda` and minimizes the same Tikhonov objective. CGLS checks a
normal-equation-residual criterion; LSQR records SciPy termination criteria.
Iteration histories are observations within a run, not independent samples.

## Evaluation and complexity

Metrics are relative reconstruction error `||x_hat-x||/||x||`, MSE, PSNR with
data range one, SSIM, residual norm `||b-Ax_hat||`, and relative residual.
Residual fit is not reconstruction accuracy. An explicit dense operator stores
`n^2` Float64 entries; separable factors store `r^2+c^2` entries plus working
arrays. These are theoretical storage estimates, not peak-RAM measurements.
