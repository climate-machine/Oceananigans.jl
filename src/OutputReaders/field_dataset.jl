struct FieldDataset{F, M, P}
      fields :: F
    metadata :: M
    filepath :: P
end

"""
    FieldDataset(filepath; architecture=CPU(), backend=InMemory(), metadata_paths=["metadata"])

Returns a `Dict` containing a `FieldTimeSeries` for each field in the JLD2 file located at `filepath`.
Note that model output must have been saved with halos. The `InMemory` backend will store the data
fully in memory as a 4D multi-dimensional array while the `OnDisk` backend will lazily load field time
snapshots when the `FieldTimeSeries` is indexed linearly.

`metadata_paths` is a list of JLD2 paths to look for metadata. By default it looks in `file["metadata"]`.
"""
function FieldDataset(filepath; architecture=CPU(), ArrayType=array_type(architecture), backend=InMemory(), metadata_paths=["metadata"])
    file = jldopen(filepath)

    field_names = keys(file["timeseries"])
    filter!(k -> k != "t", field_names)  # Time is not a field.

    ds = Dict{String, FieldTimeSeries}(
        name => FieldTimeSeries(filepath, name; architecture, ArrayType, backend)
        for name in field_names
    )

    metadata = Dict(
        k => file["$mp/$k"]
        for mp in metadata_paths
        for k in keys(file["$mp"])
    )

    close(file)

    return FieldDataset(ds, metadata, abspath(filepath))
end

Base.getindex(fds::FieldDataset, inds...) = Base.getindex(fds.fields, inds...)

Base.show(io::IO, fds::FieldDataset) where {X, Y, Z, K, A} =
    print(io, "FieldDataset with $(length(fds.fields)) fields and $(length(fds.metadata)) metadata entries.")
