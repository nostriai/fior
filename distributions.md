# Supported distributions from the exponential family

Exponential family in canonical form:

$$ p_{\boldsymbol{\eta}}(\boldsymbol{x}) = h(\boldsymbol{x}) \, e^{\boldsymbol{\eta} \cdot \boldsymbol{T}(\boldsymbol{x}) - A(\boldsymbol{\eta})} $$

For every family below, `A(η)` is the log-partition function — the value the trust layer's
`ΔF` is built from, evaluated in f64. Constants that are identical across the four `A()`
evaluations of a BMR `ΔF` may be dropped in an implementation; `distributions.md` gives the
full forms.

## Normal (scalar, independent components)

$$ \boldsymbol{T}(x) = (x, x^2)^T, \qquad \eta_1 = \mu/\sigma^2, \quad \eta_2 = -1/(2\sigma^2) $$

$$ A(\eta) = -\frac{\eta_1^2}{4 \eta_2} + \frac{1}{2} \ln\left|\frac{-1}{2 \eta_2}\right|$$

This is the `normal` family on the wire: `d` scalar Normals sharing no off-diagonal
precision. Domain: `η_2 < 0`.

## Multivariate Normal (full covariance) — `mvnormal`

$$ \boldsymbol{T}(x) = \left(x,\; -xx^\top/2\right)^\top, \qquad \eta_1 = \Lambda \mu, \quad \eta_2 = -\tfrac12 \Lambda $$

with `μ` a `d`-vector and `Λ` a symmetric positive-definite `d×d` precision matrix. The
wire layout is `η₁` (length `d`) followed by the lower triangle of the symmetric `η₂`
matrix, row-major — `d + d(d+1)/2` values in total.

$$ A(\boldsymbol{\eta}) = \tfrac12 \eta_1^\top (-\eta_2)^{-1} \eta_1 - \tfrac12 \ln \det(-2\,\eta_2) $$

equivalent, in precision form used by the reference simulation:

$$ A(\Lambda, h) = \tfrac12 h^\top \Lambda^{-1} h - \tfrac12 \ln \det \Lambda $$

up to a constant (`(d/2) ln(2π)`) that cancels identically in BMR's four-evaluation
difference. Domain: `η_2` negative definite, i.e. `Λ` positive definite (a Cholesky
factor exists).

## Gamma

$$ \boldsymbol{T}(x) = (\ln x, x)^\top, \qquad \eta_1 = \alpha - 1, \quad \eta_2 = -\beta $$

$$ A(\boldsymbol{\eta}) = \ln \Gamma(\eta_1 + 1) - (\eta_1 + 1) \ln(-\eta_2) $$

## Beta

$$ \boldsymbol{T}(x) = (\ln x, \ln(1-x))^\top, \qquad \eta_1 = a - 1, \quad \eta_2 = b - 1 $$

$$ A(\boldsymbol{\eta}) = \ln \Gamma(\eta_1 + 1) + \ln \Gamma(\eta_2 + 1) - \ln \Gamma(\eta_1 + \eta_2 + 2) $$

## Dirichlet

$$ \boldsymbol{T}(\boldsymbol{x}) = (\ln x_i \mid i \in \{1, \dots, k\})^\top, \qquad \eta_i = \alpha_i - 1 $$

$$ A(\boldsymbol{\eta}) = \sum_{i=1}^k \ln \Gamma(\eta_i + 1) - \ln \Gamma\!\left(\sum_{i=1}^k \eta_i + k\right) $$

## Categorical

$$ \boldsymbol{T}(x) = (\,[x = i] \mid i \in \{1, \dots, k\})^\top, \qquad \eta_i = \ln p_i $$

$$ A(\boldsymbol{\eta}) = \ln \sum_{i=1}^k e^{\eta_i} = \ln\!\left(1 + \sum_{i=1}^{k-1} e^{\eta_i}\right) $$

computed with the log-sum-exp trick for numerical stability.