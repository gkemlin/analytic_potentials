Julia code to generate the plots used in the counter examples sections of our
paper on the solutions to Schrödinger equations with analytic potentials,
available on [arXiv](https://arxiv.org/abs/2206.04954).

# Dependencies
Julia 1.9.3 with the following libraries:
- [DFTK.jl](https://dftk.org) 0.6.11
- PyPlot, LineSearches, LinearAlgebra, Optim
- For the linear cases : DoubleFloats and GenericLinearAlgebra

# Usage
To perform the computations, first open the Julia shell with `julia --project`
from the location of this repository and then run
```
using Pkg
Pkg.instantiate()
```
to install the required dependencies.

## Linear case with V only analytic on a finite band
To generate the plots from Section 3.4.2 of the paper, just run within the Julia
shell:
```
include("linear_egval.jl")
```
Plots are then saved in `test_decay_linear_egval.png`.

## Exponential convergence of the plane wave discretization
To generate the plots from Section 3.4.3 of the paper, just run within the Julia
shell:
```
include("pw_discretization.jl")
```
Plots are then saved in `test_pw_discretization.png`.

## Nonlinear Gross-Pitaevskii equation
To generate the plots from Section 4.2.1 of the paper, just run within the Julia
shell:
```
include("gross_pitaevskii_egval.jl")
```
Plots are then saved in `test_decay_gp.png` and `u0_gp.png`.


## Nonlinear elliptic equation
To generate the plots from Section 4.2.2 of the paper, just run within the Julia
shell:
```
include("gross_pitaevskii_source_term.jl")
```
Plots are then saved in `test_decay.png` and `u0.png`.

# Contact
This is research code, not necessarily user-friendly, actively maintened or
extremely robust. If you have questions or encounter problems, get in touch!

