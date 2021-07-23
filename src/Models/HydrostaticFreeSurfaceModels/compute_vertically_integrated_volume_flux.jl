"""
Compute the vertical integrated volume flux from the bottom to z=0 (i.e. linear free-surface)

```
U★ = ∫ᶻ Ax * u★ dz`
V★ = ∫ᶻ Ay * v★ dz`
"""
### Note - what we really want is RHS = divergence of the vertically integrated volume flux
###        we can optimize this a bit later to do this all in one go to save using intermediate variables.
function compute_vertically_integrated_volume_flux!(∫ᶻ_U, model, velocities_update_event)

    # Wait for predictor velocity update step to complete.
    wait(device(model.architecture), velocities_update_event)

    # Fill halo regions for predictor velocity.
    fill_halo_regions!(model.velocities, model.architecture, model.clock, fields(model))

    # Compute the vertically integrated volume flux
    event = launch!(model.architecture, model.grid, :xy,
                    _compute_vertically_integrated_volume_flux!,
                    ∫ᶻ_U, model.grid, model.velocities,
                    dependencies = device_event(model.architecture))
                    # include_right_boundaries = true) # <- get this to work with layout = :xy
                    
    wait(device(model.architecture), event)

    # We didn't include right boundaries, so...
    fill_halo_regions!(∫ᶻ_U, model.architecture, model.clock, fields(model))

    return nothing
end

@kernel function _compute_vertically_integrated_volume_flux!(∫ᶻ_U, grid, U)
    i, j = @index(Global, NTuple)

    @inbounds begin

        ∫ᶻ_U.u[i, j, 1] = 0
        ∫ᶻ_U.v[i, j, 1] = 0

        @unroll for k in 1:grid.Nz
            ∫ᶻ_U.u[i, j, 1] += Axᶠᶜᶜ(i, j, k, grid) * U.u[i, j, k]
            ∫ᶻ_U.v[i, j, 1] += Ayᶜᶠᶜ(i, j, k, grid) * U.v[i, j, k]
        end

     end
end
