#=
GradientDescent.jl

    Provides gradient descent methods, such as Newton and quasi-Newton methods, 
    for differentiable objective functions.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/24
=#

module GradientDescent

using LinearAlgebra, LineSearch

export
    # BFGS
    bfgs!,
    lbfgs!

# Include programs
include("bfgs.jl")
include("lbfgs.jl")

end