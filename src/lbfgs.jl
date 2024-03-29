#=
lbfgs.jl

    Quasi-Newton type update step routines using the limited-memory 
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm, that uses approximations 
    to the Hessian using past approximations as well as the gradient.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/31
=#

# Method
Base.@kwdef mutable struct LBFGS{T}
    m::T=10             # memory depth
    pseudo_iter::T=1    # pseudo iteration counter
end
# limited-memory BFGS state
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
end

"""
    twoloop!(state, m, pseudo_iter)

Compute descent direction using the two-loop recursion.

#### Arguments
  - `state::LBFGSState`     : state variables
  - `m::Integer`            : memory depth
  - `psuedo_iter::Integer`  : pseudo iteration counter   
"""
function twoloop!(state::LBFGSState, m::Integer, pseudo_iter::Integer)
    # Upper and lower bounds
    lower= max(pseudo_iter - m, 1) # project on non-negative orthant
    upper= pseudo_iter - 1

    # Initalize q at previous gradient
    copyto!(state.q, state.∇f_prev)

    # Backward pass to update q
    @inbounds for i = upper:-1:lower
        idx= mod1(i,m)  # convert index to [1,m] range

        # Store views
        s_i= view(state.s_mem, :, idx)
        y_i= view(state.y_mem, :, idx)

        state.α[idx]= state.ρ[idx] * dot(s_i, state.q)
        state.q.-= state.α[idx] .* y_i
    end

    # Initialize search direction r
    if pseudo_iter > 1
        idx= mod1(upper,m)  # convert index to [1,m] range

        # Store views
        s_i= view(state.s_mem, :, idx)
        y_i= view(state.y_mem, :, idx)

        γ= dot(s_i, y_i) * inv(sum(abs2, y_i))
        state.r.= γ .* state.q
    else
        state.r.= state.q
    end

    # Forward pass to update search direction r
    @inbounds for i = lower:upper
        idx= mod1(i,m)  # convert index to [1,m] range

        # Store views
        s_i= view(state.s_mem, :, idx)
        y_i= view(state.y_mem, :, idx)

        β= state.ρ[idx] * dot(y_i, state.r)
        state.r.+= s_i .* (state.α[idx] - β)
    end

    # Negate search direction
    rmul!(state.r, eltype(state.r)(-1))

    return nothing
end

"""
    update_state!(state, method, ls, f)

Update state using the limited-memory BFGS inverse Hessian approximation for the
search direction with an inexact backtracking linesearch, storing the result in
`x`.

#### Arguments
  - `state::LBFGSState` : state variables
  - `ls::BackTrack`     : line search parameters
  - `ls::BackTrack`     : line search parameters
  - `f::Function`       : ``f(x)``
"""
function update_state!(
    state::LBFGSState, 
    method::LBFGS, 
    ls::BackTrack, 
    f::Function
)
    # Store current state
    state.x_prev.= state.x

    # Search direction
    twoloop!(state, method.m, method.pseudo_iter)

    # Backtracking line Search
    state.f_prev= backtrack!(
        ls, 
        state.x, 
        state.x_prev, 
        state.r, 
        state.f_prev, 
        state.∇f_prev, 
        f
    )

    return nothing
end

"""
    update_memory!(state, method)

Update memory of the limited-memory BFGS algorithm to construct the inverse
Hessian approximation for the search direction.

#### Arguments
  - `state::LBFGSState` : state variables
  - `method::LBFGS`     : method info
"""
function update_memory!(state::LBFGSState, method::LBFGS)
    # Inverse curvature condition
    ρ= inv(dot(state.y, state.s))

    if isinf(ρ)
        # Erase memory, restart
        method.pseudo_iter= 1
        # Exit
        return nothing
    end

    idx= mod1(method.pseudo_iter,method.m)  # convert index to [1,m] range
    # Update memory
    state.s_mem[:,idx].= state.s
    state.y_mem[:,idx].= state.y
    state.ρ[idx]= ρ

    # Update pseudo iteration
    method.pseudo_iter+= 1

    return nothing
end

"""
    lbfgs!(method, state, ls, f, ∇f!)

Update state and inverse Hessian approximation using the limited-memory BFGS
algorithm.

#### Arguments
  - `method::LBFGS`     : method info
  - `state::LBFGSState` : state variables
  - `ls::BackTrack`     : line search parameters
  - `f::Function`       : ``f(x)``
  - `∇f!::Function`     : gradient of `f`
"""
function lbfgs!(
    method::LBFGS, 
    state::LBFGSState, 
    ls::BackTrack, 
    f::Function,            
    ∇f!::Function
)
    # Update state
    update_state!(state, method, ls, f)

    # Change in state
    state.s.= ls.α .* state.r

    # Store previous gradient
    state.y.= state.∇f_prev
    # Update gradient
    ∇f!(state.∇f_prev, state.x)
    # Change in gradient
    state.y.= state.∇f_prev .- state.y

    # Update memory
    update_memory!(state, method)

    return nothing
end