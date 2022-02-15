#=
bfgs.jl

    Quasi-Newton type update step routines using the 
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm, that uses approximations 
    to the Hessian using past approximations as well as the gradient.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/24
=#

"""
    update_state!(state, ls, f, ∇f!)

Update state using the BFGS inverse Hessian approximation for the search
direction with an inexact backtracking linesearch, storing the result in `x`.

#### Arguments
  - `state::BFGSState`  : state variables
  - `ls::BackTrack`     : line search parameters
  - `f::Function`       : ``f(x)``
  - `∇f!::Function`     : gradient of `f`
"""
function update_state!(state::BFGSState, ls::BackTrack, f::Function, ∇f!::Function)
    # Infer type
    T= eltype(state.p)

    # Current gradient
    ∇f!(state.∇f_prev, state.x)

    # Search direction
    mul!(state.p, state.Hi, state.∇f_prev, T(-1), zero(T))

    # Backtracking line Search
    f_x= backtrack!(ls, state.x, state.x_prev, state.p, state.f_prev, 
                    state.∇f_prev, f)

    # Store current objective function value
    state.f_prev= f_x

    return nothing
end

"""
    update_h!(state)

Update the inverse Hessian approximation using the BFGS algorithm, storing the
result in `Hi`.

#### Arguments
  - `state::BFGSState`  : state variables
"""
function update_h!(state::BFGSState)
    # Curvature condition
    curv= dot(state.y, state.s)

    if curv > 0
        # Cache
        mul!(state.u, state.Hi, state.y)

        # Constants
        α= (curv + dot(state.u, state.y)) * inv(curv * curv)
        β= inv(curv)

        # BFGS update, using Sherman-Morrison-Woodbury
        n= length(state.x)
        # Small dimensions: loop unfolding
        if n < 50
            for i in 1:n
                s_i= state.s[i]
                u_i= state.u[i]
                @simd for j in 1:n
                    @inbounds state.Hi[i,j]+= α * s_i * state.s[j] - 
                                                β * (u_i * state.s[j] +
                                                        state.u[j] * s_i)
                end
            end
        # Large dimensions: BLAS routines
        else
            BLAS.ger!(α, state.s, state.s, state.Hi)
            BLAS.ger!(-β, state.u, state.s, state.Hi)
            BLAS.ger!(-β, state.s, state.u, state.Hi)
        end
    end

    return nothing
end

"""
    bfgs!(state, ls, f, ∇f!)

Update state and inverse Hessian approximation using the BFGS algorithm.

#### Arguments
  - `state::BFGSState`  : state variables
  - `ls::BackTrack`     : line search parameters
  - `f::Function`       : ``f(x)``
  - `∇f!::Function`     : gradient of `f`
"""
function bfgs!(state::BFGSState, ls::BackTrack, f::Function, ∇f!::Function)
    # Update state
    update_state!(state, ls, f, ∇f!)

    # Change in state
    @. state.s= ls.α * state.p

    # Change in gradient
    ∇f!(state.y, state.x)
    @. state.y= state.y - state.∇f_prev

    # Update inverse Hessian approx
    update_h!(state)

    return nothing
end