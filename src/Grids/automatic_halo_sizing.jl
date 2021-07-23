"""
    required_halo_size(tendency_term)

Returns the required size of halos for a term appearing
in the tendency for a velocity field or tracer field.

Example
=======

```jldoctest
using Oceananigans.Advection: CenteredFourthOrder
using Oceananigans.Grids: required_halo_size

required_halo_size(CenteredFourthOrder())

# output
2
"""

function required_halo_size end

required_halo_size(nothing) = 1

inflate_halo_size_one_dimension(required_Hx, current_Hx, TX          ) = max(required_Hx, current_Hx)
inflate_halo_size_one_dimension(required_Hx, current_Hx, ::Type{Flat}) = 0

halo_size_string(Hx, TX) = "$Hx, "
halo_size_string(Hx, ::Type{Flat}) = ""
halo_size_string(H::Tuple, topo::Tuple) = "(" * prod(halo_size_str.(H, topo))[1:end-1] * ")"

function inflate_halo_size(Hx, Hy, Hz, topology, tendency_terms...)

    old_halo = (Hx, Hy, Hz)

    for term in tendency_terms
        required_H = required_halo_size(term)
        Hx = inflate_halo_size_one_dimension(required_H, Hx, topology[1])
        Hy = inflate_halo_size_one_dimension(required_H, Hy, topology[2])
        Hz = inflate_halo_size_one_dimension(required_H, Hz, topology[3])
    end

    new_halo = (Hx, Hy, Hz)
    
    if new_halo !== old_halo
        halo_str = halo_size_string(new_halo, topology)
        
        @warn "Inflating model grid halo size to $halo_str and recreating grid. " *
        "The model grid will be different from the input grid. To avoid this warning, " *
        "pass halo=$halo_str when constructing the grid."
    end

    return new_halo
end
