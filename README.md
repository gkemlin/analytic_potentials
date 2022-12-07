Julia code to generate the plots used in the counter examples sections of our
paper on the solutions to Schrödinger equations with analytic potentials,
available on [arXiv](https://arxiv.org/abs/2207.12190).

# Dependencies
Julia 1.8 with the following libraries:
- [DFTK.jl](https://dftk.org) 0.5.15
- PyPlot, LineSearches, LinearAlgebra, Optim

# Usage
To perform the computations, first open the Julia shell with `julia --project`
from the location of this repository and then run
```
using Pkg
Pkg.instantiate()
```
to install the required dependencies.

## Nonlinear Gross-Pitaevskii equation
To generate the plots from Section 4.2.1 of the paper, just run from the
location of this repository:
```
julia --project gross_pitaevskii_egval.jl
```
Plots are then saved in `test_decay_gp.png` and `u0_gp.png`.


## Nonlinear elliptic equation
To generate the plots from Section 4.2.2 of the paper, just run from the
location of this repository:
```
julia --project gross_pitaevskii_source_term.jl
```
Plots are then saved in `test_decay.png` and `u0.png`.

# Contact
This is research code, not necessarily user-friendly, actively maintened or
extremely robust. For a first contact with the implementation, looking at the
[example](https://docs.dftk.org/stable/examples/error_estimates_forces/)
mentionned above is maybe advised. If you have questions or encounter problems,
get in touch!

