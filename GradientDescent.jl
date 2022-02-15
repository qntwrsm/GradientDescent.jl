#=
GradientDescent.jl

    Provides gradient descent methods, such as Newton and quasi-Newton methods, 
    for differentiable objective functions.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/24
=#

module GradientDescent

push!(LOAD_PATH, "/Users/quintwiersma/Dropbox/VU/PhD/code/jllib/LineSearch")

using LinearAlgebra, LineSearch

export
    # BFGS
    bfgs!,
    lbfgs!

# Structs
# BFGS
mutable struct BFGSState{Tv, Tf, Tm}
    x::Tv       # current state
    x_prev::Tv  # previous state
    ∇f_prev::Tv # previous gradient of f(x)
    f_prev::Tf  # previous f(x)
    s::Tv       # change in state
    y::Tv       # change in gradient
    p::Tv       # search direction
    Hi::Tm      # inverse Hessian approx
    u::Tv       # buffer
end

# limited-memory BFGS
# Method
Base.@kwdef mutable struct LBFGS{T}
    m::T = 10           # memory depth
    pseudo_iter::T = 1  # pseudo iteration counter
end
# State
mutable struct LBFGSState{Tv, Tf, Tm}
    x::Tv       # current state
    x_prev::Tv  # previous state
    ∇f_prev::Tv # previous gradient of f(x)
    f_prev::Tf  # previous f(x)
    s::Tv       # change in state
    y::Tv       # change in gradient
    s_mem::Tm   # memory change in states
    y_mem::Tm   # memory change in gradients
    r::Tv       # search direction
    ρ::Tv       # two-loop ρ
    α::Tv       # buffer two-loop α
    q::Tv       # buffer two-loop q
    u::Tv       # buffer
end

# Include programs
include("bfgs.jl")
include("lbfgs.jl")

end