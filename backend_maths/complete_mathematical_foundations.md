# Complete Mathematical Foundations for Image Deblurring

## Purpose and learning roadmap

This guide teaches the mathematics behind the implemented numerical backend. **Core implemented mathematics** is marked by its direct connection to the repository. **Supporting theory** supplies the ideas needed to understand it. **Future extensions** are explicitly not implemented. The route is physical image formation → linear algebra → inverse-problem instability → spectral regularization → parameter selection → iterative methods → scientifically valid evaluation.

## Contents

1. [Image formation](#1-image-formation-and-degradation)
2. [Linear algebra](#2-linear-algebra-for-images)
3. [Forward model](#3-from-blur-to-the-forward-model)
4. [Vectorization and Kronecker products](#4-vectorization-and-kronecker-products)
5. [Ill-posedness](#5-inverse-problems-and-well-posedness)
6. [Conditioning](#6-conditioning-and-stability)
7. [Classical solvers](#7-classical-inverse-methods)
8. [SVD](#8-singular-value-decomposition)
9. [Picard analysis](#9-picard-analysis)
10. [Regularization theory](#10-regularization-theory)
11. [TSVD](#11-truncated-svd)
12. [Tikhonov](#12-tikhonov-regularization)
13. [Parameter selection](#13-parameter-selection)
14. [Iterative methods](#14-iterative-methods)
15. [Matrix-free computation](#15-matrix-free-and-separable-computation)
16. [Metrics](#16-evaluation-metrics)
17. [Synthesis](#17-complete-research-narrative)

## 1. Image formation and degradation

A grayscale digital image is a sampled intensity field. A continuous scene intensity is sampled on a rectangular grid to form `X in R^(r x c)`; `X[i,j]` is one pixel intensity. An optical system does not generally return that ideal sample: a point source spreads to a point-spread function (PSF), and the scene is convolved with that PSF. Noise is a separate random measurement perturbation.

For a one-dimensional Gaussian PSF, `g(t) proportional to exp(-t^2/(2 sigma^2))`. Discretizing and normalizing a finite, odd support kernel gives weights whose local average suppresses rapid spatial changes. Thus edges and texture, which have high-frequency content, are attenuated more strongly than slowly varying regions. Restoration is inverse: infer the input from an output that has lost information.

**Illustration.** For `[0,1,0]` and weights `[1/4,1/2,1/4]`, the central response is `1/2`; nearby responses receive nonzero weight. A sharp impulse is spread. Adding a perturbation such as `0.01` afterward is noise, not blur.

**Takeaway.** Blur is structured information loss; noise is uncertainty in the observation. Their combination makes naive reversal unsafe.

## 2. Linear algebra for images

A scalar is one number, a vector is an ordered list, and a matrix represents a linear map. A map `T` is linear when `T(alpha x+beta y)=alpha T(x)+beta T(y)`. Matrix multiplication is composition of maps: if `A` is `m x n`, `Ax` is defined for `x in R^n` and lies in `R^m`.

The column space contains achievable observations; the null space contains inputs mapped to zero. Rank is the dimension of the column space. Independent vectors add new directions; a basis is a minimal independent spanning set. The inner product is `x.T y`; orthogonal vectors have inner product zero, and an orthonormal basis preserves Euclidean lengths under coordinate change. Norms measure size: `||x||_2=sqrt(x.T x)` and the induced matrix norm is `||A||_2=max(||Ax||_2/||x||_2)`.

An image matrix can be represented by one long vector, and blur is a linear operator on that vector. The matrix is a representation; the underlying operation is blur itself. This distinction matters because large problems can apply the operator without storing its matrix.

## 3. From blur to the forward model

**Core implementation.** The project assumes a spatially invariant separable Gaussian blur. For normalized one-dimensional factors `B_y in R^(r x r)` and `B_x in R^(c x c)`,

```math
Y=B_yXB_x^T,\qquad b=Ax+\epsilon.
```

Rows of each factor contain shifted Gaussian weights. Values outside an image boundary are zero. When a kernel is cut by an edge, the retained weights are **not renormalized**, so boundary row sums can be less than one. This is a physical/modeling decision, not a numerical accident.

An explicit dense `A` is an `rc x rc` matrix. The implementation can instead apply the two small factors directly. Sparse Kronecker, separable, and matrix-free forms all represent the same chosen operator when their conventions agree.

## 4. Vectorization and Kronecker products

Vectorization permits image equations to use standard linear-system notation. C-order flattens rows: for `X=[[a,b,c],[d,e,f]]`, `vec_C(X)=[a,b,c,d,e,f].T`. This differs from common column-major textbook notation.

For matrices `P` and `Q`, `P kron Q` replaces every entry `p_ij` by block `p_ij Q`; if `P` is `r x r` and `Q` is `c x c`, it is `rc x rc`. The row-major identity used by this repository is

```math
\mathrm{vec}_C(B_yXB_x^T)=(B_y\otimes B_x)\mathrm{vec}_C(X).
```

It follows by applying the horizontal map inside each row and the vertical map across corresponding row positions. Reversing the factors under C-order would map components in a different order and therefore represents a different operator. This is why storage convention is mathematical, not merely programming detail.

## 5. Inverse problems and well-posedness

The forward problem computes `b` from `x`; the inverse problem estimates `x` from `b`. Hadamard calls a problem well posed when a solution exists, is unique, and depends continuously on data. Blur can violate practical stability even if a finite discrete matrix is invertible: nearby observations can require very different inverse images.

A solvable equation is not automatically a numerically stable or physically meaningful reconstruction. Nonuniqueness is related to a null space; instability is related to almost-null directions. Blur leaves some directions extremely weak, so observational noise can dominate their inferred coefficients.

## 6. Conditioning and stability

For nonsingular `A`, `kappa_2(A)=||A||_2||A^{-1}||_2=sigma_max/sigma_min`. Under small perturbations and standard assumptions,

```math
\frac{||\delta x||_2}{||x||_2}\lesssim\kappa_2(A)\frac{||\delta b||_2}{||b||_2}.
```

This is a sensitivity statement about the problem, not a guarantee about any algorithm. Algorithmic stability concerns whether a computation introduces unnecessarily large rounding errors. Floating point and measurement noise are both magnified by an ill-conditioned inverse. A blur matrix can be mathematically invertible yet computationally dangerous because `sigma_min` is tiny.

## 7. Classical inverse methods

Direct inversion uses `x=A^{-1}b`; it exists only for square nonsingular `A`, and directly exposes `1/sigma_i` amplification. Least squares solves `min_x ||Ax-b||_2^2`. Differentiating gives `A.T(Ax-b)=0`, hence normal equations `A.T A x=A.T b`; geometrically, the residual is orthogonal to the column space.

QR writes `A=QR` with orthonormal `Q` and triangular `R`, so least squares becomes `Rx=Q.Tb`. It avoids explicitly forming `A.TA`. Under full column rank,

```math
\kappa_2(A^TA)=\kappa_2(A)^2,
```

so normal equations can substantially worsen sensitivity.

The residual is `r=b-A x_hat`; reconstruction error is `x_hat-x_true`. Noise can be fitted in weak directions, producing tiny `r` and huge reconstruction error. This is exactly the Phase 2 phenomenon, not a contradiction.

## 8. Singular value decomposition

Every real `m x n` matrix has `A=U Sigma V.T`, with orthogonal singular-vector matrices and nonnegative ordered singular values. Geometrically, `V.T` rotates input coordinates, `Sigma` scales axes, and `U` rotates output coordinates. The unit sphere becomes an ellipsoid; small axes are directions nearly erased by the forward map.

For nonzero singular values, the pseudoinverse solution is

```math
A^\dagger b=\sum_i\frac{u_i^Tb}{\sigma_i}v_i.
```

The implementation reports numerical rank using a relative threshold `sigma_i > tau sigma_1`. Numerical rank is a floating-point usability decision; it is not the same as a chosen TSVD regularization rank.

## 9. Picard analysis

The SVD coefficient `u_i.Tb` says how much measured data lies along left singular vector `u_i`. In a discrete Picard picture, useful signal coefficients should decay sufficiently relative to singular values before noise dominates. The inverse coefficient `(u_i.Tb)/sigma_i` becomes unsafe when a small denominator meets noise-level numerator.

Picard plots are diagnostics, not proofs that every theoretical asymptotic condition holds. The Phase 3 implementation records coefficient, singular-value, and transformed-noise quantities to show where weak directions become noise sensitive.

## 10. Regularization theory

Regularization intentionally changes the inverse problem to obtain a stable, useful solution. A general form is

```math
\min_x ||Ax-b||_2^2+\lambda^2||Lx||_2^2.
```

`L` encodes what is penalized. **Core implementation:** `L=I`, standard L2 Tikhonov. The penalty suppresses large, noise-sensitive solutions. It introduces bias, while decreasing variance/noise amplification: the bias--variance trade-off. Total variation is a useful **future extension**, not an implemented method.

## 11. Truncated SVD

TSVD replaces the unstable full expansion by

```math
x_k=\sum_{i=1}^k\frac{u_i^Tb}{\sigma_i}v_i.
```

Its filter is hard: retained components have inverse filter `1/sigma_i`; discarded components have zero filter. Small `k` oversmooths and biases the image; large `k` admits weak-direction noise. Retaining more components is therefore not monotonic improvement. TSVD rank must not exceed the numerical rank accepted by the implementation.

## 12. Tikhonov regularization

For `L=I`, differentiate the objective to obtain

```math
(A^TA+\lambda^2I)x=A^Tb.
```

Substitution of the SVD gives

```math
x_\lambda=\sum_i\frac{\sigma_i}{\sigma_i^2+\lambda^2}(u_i^Tb)v_i.
```

The dimensionless retention factor is `sigma_i^2/(sigma_i^2+lambda^2)`: unlike TSVD it shrinks smoothly. As `lambda` approaches zero, the solution approaches the unregularized inverse where defined; large `lambda` approaches zero. The equivalent augmented least-squares system is `[A; lambda I]x=[b;0]`, which avoids explicitly forming normal equations. Both forms are implemented and compared.

## 13. Parameter selection

Choosing `k` or `lambda` is itself an inverse-problem decision. The L-curve plots `log ||Ax-b||` against `log ||x||`; the implementation estimates finite discrete curvature and selects its largest finite value. A corner is a heuristic balance, not a theorem of optimality.

For effective degrees of freedom `d`, implemented GCV is

```math
G=\frac{||Ax-b||_2^2/n}{(1-d/n)^2}.
```

For TSVD, `d=k`; for Tikhonov it is the sum of retention factors. GCV estimates predictive performance but can select a poor finite-noise candidate.

The discrepancy target is `eta sigma_epsilon sqrt(n)`, with `eta=1.05` in saved experiments. The candidate residual nearest the target is selected; if the target is outside the grid range, an endpoint is selected and documented. Ground truth is used only after selection for evaluation. Selecting by ground-truth error would be oracle tuning, not observation-only selection.

## 14. Iterative methods

Direct dense factorizations become expensive for large `n`; iterative solvers require only `A v` and `A.T w`. LSQR is based on Golub--Kahan bidiagonalization and solves least squares without explicitly constructing `A.TA`. It is closely related to conjugate-gradient ideas on normal equations.

CGLS starts with `x_0=0`, residual `r_0=b-Ax_0`, gradient `g_0=A.Tr_0`, and search direction `p_0=g_0`. A step uses `alpha=(g.Tg)/||Ap||^2`, updates `x` and `r`, then uses `beta=(g_new.Tg_new)/(g.Tg)` to form the next conjugate direction. The implementation checks the normal-equation residual.

Damped LSQR minimizes `||Ax-b||^2+lambda^2||x||^2`; its SciPy `damp` is exactly project `lambda`, not `lambda^2`. Iteration count is an implicit regularization parameter. Semiconvergence occurs when early iterations recover stable components and later iterations begin fitting noise: residual may fall while reconstruction error rises. More iterations are not automatically better.

## 15. Matrix-free and separable computation

An explicit dense operator needs `(rc)^2` Float64 entries. A separable representation stores `r^2+c^2` factor entries and applies `B_y X B_x.T`; a `LinearOperator` exposes this action and its exact adjoint without building full `A`. Sparse Kronecker storage is another explicit representation.

Matrix-free does not mean zero memory or optimal runtime. The implementation uses dense one-dimensional factors, so matrix products and temporary `r x c` arrays still cost work and storage. Reported Phase 7 dense-versus-separable values are theoretical array-storage estimates, not measured peak RAM.

## 16. Evaluation metrics

Relative reconstruction error is `||x_hat-x||_2/||x||_2`; MSE is the mean squared pixel difference; PSNR is `10 log10(1/MSE)` for normalized range one; SSIM compares local luminance, contrast, and structure. Residual norm is `||b-Ax_hat||_2`, and relative residual divides it by `||b||_2` where nonzero.

No metric is sufficient alone. MSE and relative error require a reference, residual measures data fit, and SSIM is perceptual/structural rather than a physical inverse-model criterion. Raw reconstructions are deliberately not clipped before metric computation, so instability remains visible instead of cosmetically hidden.

## 17. Complete research narrative

Imaging creates blur and noise; discretization produces `b=Ax+epsilon`; ill-conditioning makes direct inversion unstable; SVD identifies weak directions; TSVD and Tikhonov filter them; parameter rules choose filter strength without ground-truth leakage; matrix-free iterative solvers extend the study to larger images but introduce stopping and semiconvergence; metrics and Phase 8 compatibility controls prevent invalid conclusions.

| Approach | Main strength | Main limitation |
|---|---|---|
| Classical inverse/least squares | Direct reference baseline | Amplifies weak-direction noise |
| TSVD | Transparent hard spectral filter | Rank choice is sensitive |
| Tikhonov | Smooth controllable filtering | Lambda choice and bias matter |
| Iterative regularization | Matrix-free scalability | Stopping can determine quality |

No approach universally dominates. The right interpretation is conditional on operator, noise, configuration, parameter rule, and evaluation target.

## Glossary and notation

`A`: forward blur operator; `x`: sharp-image vector; `b`: observation; `epsilon`: noise; `X,Y`: image matrices; `B_y,B_x`: vertical/horizontal factors; `U,Sigma,V`: SVD factors; `sigma_i`: singular value; `k`: TSVD rank; `lambda`: Tikhonov/damping parameter; `r`: residual; `n=rc`: pixel/vector count; `kappa_2`: spectral condition number; `vec_C`: C-order vectorization; adjoint: transpose action under the Euclidean inner product; semiconvergence: later iterative degradation after early improvement.
