using StaticArrays
using SymPy

"""
WENO reconstruction schemes following the lecture notes by Shu (1998) and
the review article by Shu (2009). WENO smoothness indicators are due to
Jiang & Shu (1996).

Shu (1998), "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
    Schemes for Hyperbolic Conservation Laws", In: Advanced Numerical Approximation
    of Nonlinear Hyperbolic Equations, edited by Cockburn et al., pp. 325–432.
    DOI: https://doi.org/10.1007/BFb0096355

Shu (2009) "High Order Weighted Essentially Nonoscillatory Schemes for Convection
    Dominated Problems", SIAM Review 51(1), pp. 82–126.
    DOI: https://doi.org/10.1137/070679065

Jiang & Shu (1996) "Efficient Implementation of Weighted ENO Schemes", Journal of
    Computational Physics 126, pp. 202–228.
    DOI: https://doi.org/10.1006/jcph.1996.0130
"""

#####
##### ENO reconstruction coefficients and WENO weights on a uniform grid
##### Equation (2.21) from the Shu (1998) lecture notes.
#####

"""
    UΠ(k, r, m, l)

Return the product in the numerator of equation (2.21) of Shu (1998) for the `l`th
term of the `m`th Lagrange basis polynomial of an ENO reconstruction scheme with
order `k` and left shift `r`.
"""
UΠ(k, r, l, m) = prod([r-q+1 for q in 0:k if q ∉ (m, l)])

"""
    U(k, r, m)

Return the numerator in equation (2.21) of Shu (1998) for the `m`th Lagrange basis
polynomial of an ENO reconstruction scheme with order `k` and left shift `r`.
"""
U(k, r, m) = sum([UΠ(k, r, m, l) for l in 0:k if l != m])

"""
    D(k, m)

Return the denominator in equation (2.21) of Shu (1998) for the `m`th Lagrange basis
polynomial of an ENO reconstruction scheme with order `k`.
"""
D(k, m) = prod([m - l for l in 0:k if l != m])

"""
    eno_coefficient(k, r)

Return the `j`th ENO coefficient used to reconstruct a value at the point x(i+½) with
order of accuracy `k` (stencil size) and left shift `r`.
"""
eno_coefficient(k, r, j) = sum([U(k, r, m)//D(k, m) for m in j+1:k])

"""
    eno_coefficients(k, r)

Return an array of ENO coefficients to reconstruct a value at the point x(i+½) with
order of accuracy `k` (stencil size) and left shift `r`.
"""
eno_coefficients(k, r) = [eno_coefficient(k, r, j) for j in 0:k-1]

"""
    eno_coefficients_matrix([FT=Rational], k)

Return a k×k static array containing ENO coefficients to reconstruct a value at the
point x(i+½) with order of accuracy `k` (stencil size) with element type `FT`. Note
that when combined these ENO interpolants produce a WENO scheme of order 2k-1.
"""
eno_coefficients_matrix(FT, k) =
    cat([eno_coefficients(k, r) for r in 0:k-1]..., dims=1) |> SMatrix{k,k,FT}

eno_coefficients_matrix(k) = eno_coefficients_matrix(Rational, k)

"""
    optimal_weno_weights([FT=Rational], k)

Return a static vector containing the optimal weights that can be used to weigh ENO
reconstruction schemes of order `k` to produce a WENO scheme of order 2k-1.
"""
function optimal_weno_weights(FT, k)
    C = zeros(Rational, 2k-1, k)
    b = eno_coefficients(2k-1, k-1)

    for n in 0:k-1
        C[n+1:n+k, n+1] .= eno_coefficients(k, k-1-n)
    end

    γ = C \ b
    γ = rationalize.(γ, tol=√eps(Float64)) |> reverse
    return SVector{k,FT}(γ)
end

optimal_weno_weights(k) = optimal_weno_weights(Rational, k)

#####
##### Jiang & Shu (1996) WENO smoothness indicators β
##### See equation (2.61) of Shu (1998).
#####

x(j) = j  # Assume uniform grid for now.

"""
    ℓ(ξ, k, r, j)

Return a symbolic expression for the `j`th Lagrange basis polynomial ℓ(ξ) for an
ENO reconstruction scheme of order k and left shift `r` with field values `ϕ`.
"""
ℓ(ξ, k, r, j) = prod((ξ - x(m-r)) / (x(j-r) - x(m-r)) for m in 0:k-1 if m != j)

"""
    L(ξ, k, r, ϕ)

Return a symbolic expression for the interpolating Lagrange polynomial p(ξ) for an
ENO reconstruction scheme of order k and left shift `r` with field values `ϕ`.
"""
p(ξ, k, r, ϕ) = sum(ℓ(ξ, k, r, j) * ϕ[j+1] for j in 0:k-1)

"""
    β(k, r, ϕ)

Return a symbolic expression for the smoothness indicator β for a WENO reconstruction
scheme with stencils of size `k` and left shift `r` (WENO scheme of order 2k-1). The
field values are represented by the symbols in `ϕ` which should have length k.
"""
function β(k, r, ϕ)
    @vars ξ
    return sum(integrate(diff(p(ξ, k, r, ϕ), ξ, l)^2, (ξ, Sym(-1//2), Sym(1//2))) for l in 1:k-1)
end

"""
    subscript(n)

Convert the integer `n` to a subscript in the form of a unicode string. Note
that `0x2080` is the unicode encoding for the subscript 0.
"""
subscript(n) = join(Char(0x2080 + parse(Int, d)) for d in string(n))

subscript_sign(n) = n > 0 ? "₊" : n < 0 ? "₋" : ""

subscript_index(n) = n == 0 ? "" : subscript_sign(n) * subscript(abs(n))

"""
    β_coefficients([FT=Rational], k)

Return a k×k×k static array containing the WENO smoothness indicator coefficients
described by Jiang & Shu (1998) for a WENO reconstruction with stencils of size `k`
(WENO scheme of order 2k-1) with element type `FT`.

The `B[m, n, r]` coefficient corresponds to the coefficient of the
`ϕ[r-k+m+1] * ϕ[r-k+n+1]` term where r-k+1 <= m, n <= r and `r` is the left shift of
the ENO interpolants.
"""
function β_coefficients(FT, k)
    B = zeros(Float64, k, k, k)

    for r in 0:k-1
        ϕ = [Sym("ϕᵢ" * subscript_index(n)) for n in r:-1:r-k+1]
        β_symbolic = β(k, r, ϕ) |> expand

        for m in 1:k, n in 1:k
            B[m, n, r+1] = β_symbolic.coeff(ϕ[m] * ϕ[n])
        end
    end

    B = rationalize.(B, tol=√eps(Float64))
    return SArray{Tuple{k,k,k},FT}(B)
end

β_coefficients(k) = β_coefficients(Rational, k)
