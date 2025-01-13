import OnlineStats:
    OnlineStats, Group, Moments, fit!, OnlineStat, Weight, EqualWeight, mean, std
export scalar_normalizer, array_normalizer, NormalizedTraces, Normalizer
import MacroTools.@forward

"""
    Normalizer(::OnlineStat)

Wraps an OnlineStat to be used by a [`NormalizedTraces`](@ref).
"""
struct Normalizer{OS<:OnlineStat}
    os::OS
end

@forward Normalizer.os OnlineStats.mean,
OnlineStats.std,
Base.iterate,
normalize,
Base.length

#Treats last dim as batch dim
function OnlineStats.fit!(n::Normalizer, data::AbstractArray)
    for d in eachslice(data, dims = ndims(data))
        fit!(n.os, vec(d))
    end
    n
end

function OnlineStats.fit!(n::Normalizer{<:Group}, y::AbstractVector)
    fit!(n.os, y)
    n
end

function OnlineStats.fit!(n::Normalizer, y)
    for yi in y
        fit!(n.os, vec(yi))
    end
    n
end

function OnlineStats.fit!(n::Normalizer{<:Moments}, y::AbstractVector{<:Number})
    for yi in y
        fit!(n.os, yi)
    end
    n
end

function OnlineStats.fit!(n::Normalizer, data::Number)
    fit!(n.os, data)
    n
end

"""
    normalize(os::Moments, x)

Given an Moments estimate of the elements of x, a vector of scalar traces,
normalizes x elementwise to zero mean, and unit variance. 
"""
function normalize(os::Moments, x)
    T = eltype(x)
    m, s = T(mean(os)), T(std(os))
    return (x .- m) ./ s
end

"""
    normalize(os::Group{<:AbstractVector{<:Moments}}, x)

Given an os::Group{<:Tuple{Moments}}, that is, a multivariate estimator of the moments of 
each element of x,
normalizes each element of x to zero mean, and unit variance. Treats the last dimension as 
a batch dimension if `ndims(x) >= 2`.
"""
function normalize(os::Group{<:AbstractVector{<:Moments}}, x::AbstractVector)
    T = eltype(x)
    m = [T(mean(stat)) for stat in os]
    s = [T(std(stat)) for stat in os]
    return (x .- m) ./ s
end

function normalize(os::Group{<:AbstractVector{<:Moments}}, x::AbstractArray)
    xn = similar(x)
    for (i, slice) in enumerate(eachslice(x, dims = ndims(x)))
        xn[repeat([:], ndims(x) - 1)..., i] .=
            reshape(normalize(os, vec(slice)), size(x)[1:end-1]...)
    end
    return xn
end

function normalize(
    os::Group{<:AbstractVector{<:Moments}},
    x::AbstractVector{<:AbstractArray},
)
    xn = similar(x)
    for (i, el) in enumerate(x)
        xn[i] = normalize(os, vec(el))
    end
    return xn
end

"""
    scalar_normalizer(;weights = OnlineStats.EqualWeight())

Returns preconfigured normalizer for scalar traces such as rewards. By default, all samples 
have equal weights in the computation of the moments.
See the [OnlineStats documentation](https://joshday.github.io/OnlineStats.jl/stable/weights/) 
to use variants such as exponential weights to favor the most recent observations.
"""
scalar_normalizer(; weight::Weight = EqualWeight()) = Normalizer(Moments(weight = weight))

"""
    array_normalizer(size::Tuple{Int}; weights = OnlineStats.EqualWeight())

Returns preconfigured normalizer for array traces such as vector or matrix states. 
`size` is a tuple containing the dimension sizes of a state. E.g. `(10,)` for a 10-elements
vector, or `(252,252)` for a square image.
By default, all samples have equal weights in the computation of the moments.
See the [OnlineStats documentation](https://joshday.github.io/OnlineStats.jl/stable/weights/) 
to use variants such as exponential weights to favor the most recent observations.
"""
array_normalizer(size::NTuple{N,Int}; weight::Weight = EqualWeight()) where {N} =
    Normalizer(Group([Moments(weight = weight) for _ = 1:prod(size)]))

"""
    NormalizedTraces(traces::AbstractTraces, normalizers::NamedTuple)
    NormalizedTraces(traces::AbstractTraces; trace_normalizer_pairs...)

Wraps an [`AbstractTraces`](@ref) and a `NamedTuple` of `Symbol` => [`Normalizer`](@ref) 
pairs. 
When pushing new elements to the traces, a `NormalizedTraces` will first update a running 
estimate of the moments of traces present in the keys of `normalizers`.
When sampling a normalized trace, it will first normalize the samples to zero mean and unit 
variance. Traces that do not have a normalizer are sample as usual.

Note that when used in combination with [`Episodes`](@ref), `NormalizedTraces` must wrap 
the `Episodes` struct, not the inner `AbstractTraces` contained in an `Episode`, otherwise 
the running estimate will reset after each episode.

When used with a MultiplexTraces, the normalizer used with for one symbol (e.g. :state) will
be the same used for the other one (e.g. :next_state).

Preconfigured normalizers are provided for scalar (see [`scalar_normalizer`](@ref)) and 
arrays (see [`array_normalizer`](@ref)).

# Examples
```
t = CircularArraySARTSTraces(capacity = 10, state = Float64 => (5,))
nt = NormalizedTraces(t, reward = scalar_normalizer(), state = array_normalizer((5,)))
# :next_state will also be normalized. 
traj = Trajectory(
    container = nt,
    sampler = BatchSampler(10)
)
```
"""
struct NormalizedTraces{names,TT,T<:AbstractTraces{names,TT},normnames,N} <:
       AbstractTraces{names,TT}
    traces::T
    normalizers::NamedTuple{normnames,N}
end

function NormalizedTraces(
    traces::AbstractTraces{names,TT};
    trace_normalizer_pairs...,
) where {names} where {TT}
    for key in keys(trace_normalizer_pairs)
        @assert key in keys(traces) "Traces do not have key $key, valid keys are $(keys(traces))."
    end
    nt = (; trace_normalizer_pairs...)
    for trace in traces.traces
        #check if all traces of MultiplexTraces are in pairs
        if trace isa MultiplexTraces
            if length(intersect(keys(trace), keys(trace_normalizer_pairs))) in
               [0, length(keys(trace))] #check if none or all keys are in normalizers
                continue
            else #if not then one is missing
                present_key = only(intersect(keys(trace), keys(trace_normalizer_pairs)))
                absent_key = only(setdiff(keys(trace), keys(trace_normalizer_pairs)))
                nt = merge(nt, (; (absent_key => nt[present_key],)...)) #assign the same normalizer
            end
        end
    end
    NormalizedTraces{names,TT,typeof(traces),keys(nt),typeof(values(nt))}(traces, nt)
end

function Base.show(io::IO, ::MIME"text/plain", t::NormalizedTraces{names,T}) where {names,T}
    s = nameof(typeof(t))
    println(io, "$s with $(length(names)) entries:")
    for n in names
        print(io, "  :$n => $(summary(t[n]))")
        if n in keys(t.normalizers)
            println(io, " => Normalized")
        else
            println(io, "")
        end
    end
end

@forward NormalizedTraces.traces Base.length,
Base.size,
Base.lastindex,
Base.firstindex,
Base.view,
Base.pop!,
Base.popfirst!,
Base.empty!,
Base.parent

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(nt::NormalizedTraces, x::T) where {T}
        for key in intersect(keys(nt.normalizers), fieldnames(T))
            fit!(nt.normalizers[key], getfield(x, key))
        end
        $f(nt.traces, x)
    end
end

function StatsBase.sample(
    s::BatchSampler,
    nt::NormalizedTraces,
    names,
    weights = StatsBase.UnitWeights{Int}(length(nt)),
)
    inds = StatsBase.sample(s.rng, 1:length(nt), weights, s.batchsize)
    maybe_normalize(data, key) =
        key in keys(nt.normalizers) ? normalize(nt.normalizers[key], data) : data
    NamedTuple{names}(collect(maybe_normalize(nt[x][inds], x)) for x in names)
end

function Base.getindex(nt::NormalizedTraces, inds)
    maybe_normalize(data, key) =
        key in keys(nt.normalizers) ? normalize(nt.normalizers[key], data) : data
    NamedTuple{keys(nt.traces)}(
        collect(maybe_normalize(nt.traces[x][inds], x)) for x in keys(nt.traces)
    )
end

function Base.getindex(nt::NormalizedTraces, s::Symbol)
    getindex(nt.traces, s)
end

ispartial_insert(traces::NormalizedTraces, xs) = ispartial_insert(traces.traces, xs)
