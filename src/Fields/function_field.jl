import Oceananigans.Architectures: architecture

"""
    FunctionField{X, Y, Z, C, F, G} <: AbstractField{X, Y, Z, F, G}

An `AbstractField` that returns a function evaluated at location `(X, Y, Z)` (and time, if
`C` is not `Nothing`) when indexed at `i, j, k`.
"""
struct FunctionField{X, Y, Z, C, P, F, G} <: AbstractField{X, Y, Z, F, G}
          func :: F
          grid :: G
         clock :: C
    parameters :: P

    """
        FunctionField{X, Y, Z}(func, grid; clock=nothing, parameters=nothing) where {X, Y, Z}

    Returns a `FunctionField` on `grid` and at location `X, Y, Z`.

    If `clock` is not specified, then `func` must be a function with signature
    `func(x, y, z)`. If clock is specified, `func` must be a function with signature
    `func(x, y, z, t)`, where `t` is internally determined from `clock.time`.

    A FunctionField will return the result of `func(x, y, z [, t])` at `X, Y, Z` on
    `grid` when indexed at `i, j, k`.
    """
    function FunctionField{X, Y, Z}(func, grid; clock=nothing, parameters=nothing) where {X, Y, Z}
        return new{X, Y, Z, typeof(clock),
                   typeof(parameters), typeof(func), typeof(grid)}(func, grid, clock, parameters)
    end

    """
        FunctionField{X, Y, Z}(func::FunctionField, grid; clock) where {X, Y, Z}

    Adds `clock` to an existing `FunctionField` and relocates it to `(X, Y, Z)` on `grid`.
    """
    function FunctionField{X, Y, Z}(f::FunctionField, grid; clock=nothing) where {X, Y, Z}
        return new{X, Y, Z, typeof(clock),
                   typeof(f.parameters), typeof(f.func), typeof(grid)}(f.func, grid, clock, f.parameters)
    end
end

"""Return `a`, or convert `a` to `FunctionField` if `a::Function`"""
fieldify(L, a, grid) = a
fieldify(L, a::Function, grid) = FunctionField(L, a, grid)

"""
    FunctionField(L::Tuple, func, grid)

Returns a stationary `FunctionField` on `grid` and at location `L = (X, Y, Z)`,
where `func` is callable with signature `func(x, y, z)`.
"""
FunctionField(L::Tuple, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)

# Ordinary functions needed for fields
architecture(::FunctionField) = nothing
Base.parent(f::FunctionField) = f

# Various possibilities
@inline call_func(clock, parameters, func, x, y, z)     = func(x, y, z, clock.time, parameters)
@inline call_func(::Nothing, parameters, func, x, y, z) = func(x, y, z, parameters)
@inline call_func(clock, ::Nothing, func, x, y, z)      = func(x, y, z, clock.time)
@inline call_func(::Nothing, ::Nothing, func, x, y, z)  = func(x, y, z)

@inline Base.getindex(f::FunctionField{X, Y, Z}, i, j, k) where {X, Y, Z} =
    call_func(f.clock, f.parameters, f.func,
              xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid))

@inline (f::FunctionField)(x, y, z) = call_func(f.clock, f.parameters, f.func, x, y, z)

# set! for function fields
set!(u, f::FunctionField) = set!(u, (x, y, z) -> f.func(x, y, z, f.clock.time, f.parameters))
set!(u, f::FunctionField{X, Y, Z, <:Nothing}) where {X, Y, Z} = set!(u, (x, y, z) -> f.func(x, y, z, f.parameters))
set!(u, f::FunctionField{X, Y, Z, C, <:Nothing}) where {X, Y, Z, C} = set!(u, (x, y, z) -> f.func(x, y, z, f.clock.time))
set!(u, f::FunctionField{X, Y, Z, <:Nothing, <:Nothing}) where {X, Y, Z} = set!(u, (x, y, z) -> f.func(x, y, z))

Adapt.adapt_structure(to, f::FunctionField{X, Y, Z}) where {X, Y, Z} =
    FunctionField{X, Y, Z}(Adapt.adapt(to, f.func),
                           f.grid,
                           clock = Adapt.adapt(to, f.clock),
                           parameters = Adapt.adapt(to, f.parameters))

Base.show(io::IO, field::FunctionField) =
    print(io, "FunctionField located at ", show_location(field), '\n',
          "├── func: $(short_show(field.func))", '\n',
          "├── grid: $(short_show(field.grid))\n",
          "├── clock: $(short_show(field.clock))\n",
          "└── parameters: $(field.parameters)")
