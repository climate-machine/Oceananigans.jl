using Adapt

struct Field{X, Y, Z, A, D, G, T, B} <: AbstractDataField{X, Y, Z, A, G, T, 3, D}
                   data :: D
           architecture :: A
                   grid :: G
    boundary_conditions :: B

    function Field{X, Y, Z}(data::D, arch::A, grid::G, bcs::B) where {X, Y, Z, D, A, G, B}
        T = eltype(grid)
        return new{X, Y, Z, A, D, G, T, B}(data, arch, grid, bcs)
    end
end

"""
    Field(X, Y, Z, [arch = CPU()], grid,
          [ bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
           data = new_data(eltype(grid), arch, grid, (X, Y, Z))])

Construct a `Field` on `grid` with `data` on architecture `arch` with
boundary conditions `bcs`. Each of `(X, Y, Z)` is either `Center` or `Face` and determines
the field's location in `(x, y, z)`.

Example
=======

```jldoctest
julia> using Oceananigans, Oceananigans.Grids

julia> ω = Field(Face, Face, Center, CPU(), RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))
Field located at (Face, Face, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (1, 1, 1)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
└── boundary conditions: x=(west=Periodic, east=Periodic), y=(south=Periodic, north=Periodic), z=(bottom=ZeroFlux, top=ZeroFlux)
```
"""
function Field(X, Y, Z,
               arch::AbstractArchitecture,
               grid::AbstractGrid,
               bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
               data = new_data(eltype(grid), arch, grid, (X, Y, Z)))

    validate_field_data(X, Y, Z, data, grid)

    return Field{X, Y, Z}(data, arch, grid, bcs)
end

# Default CPU architecture
Field(X, Y, Z, grid::AbstractGrid, args...) = Field(X, Y, Z, CPU(), grid, args...)

# Canonical `similar` for AbstractField (doesn't transfer boundary conditions)
Base.similar(f::AbstractField{X, Y, Z, Arch}) where {X, Y, Z, Arch} = Field(X, Y, Z, Arch(), f.grid)

# Type "destantiation": convert Face() to Face and Center() to Center if needed.
destantiate(X) = typeof(X)
destantiate(X::DataType) = X

"""
    Field(L::Tuple, arch, grid, data, bcs)

Construct a `Field` at the location defined by the 3-tuple `L`,
whose elements are `Center` or `Face`.
"""
Field(L::Tuple, args...) = Field(destantiate.(L)..., args...)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CenterField([arch=CPU()], grid
                bcs = TracerBoundaryConditions(grid),
                data = new_data(eltype(grid), arch, grid, (Center, Center, Center)))

Return a `Field{Center, Center, Center}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function CenterField(arch::AbstractArchitecture,
                     grid,
                     bcs = TracerBoundaryConditions(grid),
                     data = new_data(eltype(grid), arch, grid, (Center, Center, Center)))

    return Field(Center, Center, Center, arch, grid, bcs, data)
end

"""
    XFaceField([arch=CPU()], grid
               bcs = UVelocityBoundaryConditions(grid),
               data = new_data(eltype(grid), arch, grid, (Face, Center, Center)))

Return a `Field{Face, Center, Center}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function XFaceField(arch::AbstractArchitecture,
                    grid,
                    bcs = UVelocityBoundaryConditions(grid),
                    data = new_data(eltype(grid), arch, grid, (Face, Center, Center)))

    return Field(Face, Center, Center, arch, grid, bcs, data)
end

"""
    YFaceField([arch=CPU()], grid,
               bcs = VVelocityBoundaryConditions(grid),
               data = new_data(eltype(grid), arch, grid, (Center, Face, Center)))

Return a `Field{Center, Face, Center}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function YFaceField(arch::AbstractArchitecture,
                    grid,
                    bcs = VVelocityBoundaryConditions(grid),
                    data = new_data(eltype(grid), arch, grid, (Center, Face, Center)))

    return Field(Center, Face, Center, arch, grid, bcs, data)
end

"""
    ZFaceField([arch=CPU()], grid
               bcs = WVelocityBoundaryConditions(grid),
               data = new_data(eltype(grid), arch, grid, (Center, Center, Face)))

Return a `Field{Center, Center, Face}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function ZFaceField(arch::AbstractArchitecture,
                    grid,
                    bcs = WVelocityBoundaryConditions(grid),
                    data = new_data(eltype(grid), arch, grid, (Center, Center, Face)))

    return Field(Center, Center, Face, arch, grid, bcs, data)
end

# Default CPU architectures...
CenterField(grid::AbstractGrid, args...) = CenterField(CPU(), grid, args...)
 XFaceField(grid::AbstractGrid, args...) =  XFaceField(CPU(), grid, args...)
 YFaceField(grid::AbstractGrid, args...) =  YFaceField(CPU(), grid, args...)
 ZFaceField(grid::AbstractGrid, args...) =  ZFaceField(CPU(), grid, args...)

Adapt.adapt_structure(to, field::Field) = Adapt.adapt(to, field.data)
