
struct FunctionField{X, Y, Z, C, P, F, G, T} <: AbstractField{X, Y, Z, Nothing, G, T, 3, Nothing}
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
    function FunctionField{X, Y, Z}(func::F, grid::G; clock::C=nothing, parameters::P=nothing) where {X, Y, Z, F, G, C, P}
        T = eltype(grid)
        return new{X, Y, Z, C, P, F, G, T}(func, grid, clock, parameters)
    end

    """
        FunctionField{X, Y, Z}(func::FunctionField, grid; clock) where {X, Y, Z}

    Adds `clock` to an existing `FunctionField` and relocates it to `(X, Y, Z)` on `grid`.
    """
    function FunctionField{X, Y, Z}(f::FunctionField, grid::G; clock::C=nothing) where {X, Y, Z, G, C}
        P = typeof(f.parameters)
        T = eltype(grid)
        F = typeof(f.func)
        return new{X, Y, Z, C, P, F, G, T}(f.func, grid, clock, f.parameters)
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
architecture(f::FunctionField) = nothing
Base.parent(f::FunctionField) = f

# Various possibilities for calling FunctionField.func:
@inline call_func(clock, parameters, func, x, y, z)     = func(x, y, z, clock.time, parameters)
@inline call_func(::Nothing, parameters, func, x, y, z) = func(x, y, z, parameters)
@inline call_func(clock, ::Nothing, func, x, y, z)      = func(x, y, z, clock.time)
@inline call_func(::Nothing, ::Nothing, func, x, y, z)  = func(x, y, z)

# For setting ReducedField
@inline call_func(::Nothing, ::Nothing, func, x, y)     = func(x, y)
@inline call_func(::Nothing, ::Nothing, func, x)        = func(x)

@inline Base.getindex(f::FunctionField{LX, LY, LZ}, i::Integer, j::Integer, k::Integer) where {LX, LY, LZ} =
    call_func(f.clock, f.parameters, f.func, node(LX(), LY(), LZ(), i, j, k, f.grid)...)

@inline (f::FunctionField)(x...) = call_func(f.clock, f.parameters, f.func, x...)

Adapt.adapt_structure(to, f::FunctionField{X, Y, Z}) where {X, Y, Z} =
    FunctionField{X, Y, Z}(Adapt.adapt(to, f.func),
                           Adapt.adapt(to, f.grid),
                           clock = Adapt.adapt(to, f.clock),
                           parameters = Adapt.adapt(to, f.parameters))

Base.show(io::IO, field::FunctionField) =
    print(io, "FunctionField located at ", show_location(field), '\n',
          "├── func: $(short_show(field.func))", '\n',
          "├── grid: $(short_show(field.grid))\n",
          "├── clock: $(short_show(field.clock))\n",
          "└── parameters: $(field.parameters)")
