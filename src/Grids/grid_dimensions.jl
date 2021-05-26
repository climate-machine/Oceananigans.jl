using DimensionalData: Sampled, Ordered, Regular, Points

import DimensionalData

function DimensionalData.dims(grid::RegularRectilinearGrid, loc)
    x_dim = DimensionalData.X(xnodes(loc[1], grid), mode=Sampled(order=Ordered(), span=Regular(grid.Δx), sampling=Points()))
    y_dim = DimensionalData.Y(xnodes(loc[2], grid), mode=Sampled(order=Ordered(), span=Regular(grid.Δy), sampling=Points()))
    z_dim = DimensionalData.Z(xnodes(loc[3], grid), mode=Sampled(order=Ordered(), span=Regular(grid.Δz), sampling=Points()))
    return (x_dim, y_dim, z_dim)
end
