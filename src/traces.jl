export Trace, Traces, MultiplexTraces

import MacroTools: @forward

import CircularArrayBuffers.CircularArrayBuffer
using ElasticArrays: ElasticArray
import Adapt

#####

abstract type AbstractTrace{E} <: AbstractVector{E} end

Base.convert(::Type{AbstractTrace}, x::AbstractTrace) = x

Base.summary(io::IO, t::AbstractTrace) =
    print(io, "$(length(t))-element $(nameof(typeof(t)))")

#####

"""
    Trace(A::AbstractArray)

Similar to
[`Slices`](https://github.com/JuliaLang/julia/blob/master/base/slicearray.jl)
which will be introduced in `Julia@v1.9`. The main difference is that, the
`axes` info in the `Slices` is static, while it may be dynamic with `Trace`.

We only support slices along the last dimension since it's the most common usage
in RL.
"""
struct Trace{T,E} <: AbstractTrace{E}
    parent::T
end

Base.summary(io::IO, t::Trace{T}) where {T} =
    print(io, "$(length(t))-element$(length(t) > 0 ? 's' : "") $(nameof(typeof(t))){$T}")

function Trace(x::T) where {T<:AbstractArray}
    E = eltype(x)
    N = ndims(x) - 1
    P = typeof(x)
    I = Tuple{ntuple(_ -> Base.Slice{Base.OneTo{Int}}, Val(ndims(x) - 1))...,Int}
    Trace{T,SubArray{E,N,P,I,true}}(x)
end

Adapt.adapt_structure(to, t::Trace) = Trace(Adapt.adapt_structure(to, t.parent))

Base.convert(::Type{AbstractTrace}, x::AbstractArray) = Trace(x)

Base.size(x::Trace) = (size(x.parent, ndims(x.parent)),)
Base.getindex(s::Trace, I) = Base.maybeview(
    s.parent,
    ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...,
)
Base.setindex!(s::Trace, v, I) = setindex!(
    s.parent,
    v,
    ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...,
)

@forward Trace.parent Base.parent,
Base.pushfirst!,
Base.push!,
Base.append!,
Base.prepend!,
Base.pop!,
Base.popfirst!,
Base.empty!,
Base.eltype

#By default, AbstractTrace have infinity capacity (like a Vector). This method is specialized for 
#CircularArraySARTSTraces in common.jl. The functions below are made that way to avoid type piracy.
capacity(t::AbstractTrace) = ReinforcementLearningTrajectories.capacity(t.parent)
capacity(t::CircularArrayBuffer) = CircularArrayBuffers.capacity(t)
capacity(::AbstractVector) = Inf
capacity(::ElasticArray) = Inf

#####

"""
For each concrete `AbstractTraces`, we have the following assumption:

1. Every inner trace is an `AbstractVector`
1. Support partial updating
1. Return *View* by default when getting elements.
"""
abstract type AbstractTraces{names,T} <: AbstractVector{NamedTuple{names,T}} end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractTraces{names,T}) where {names,T}
    s = nameof(typeof(t))
    println(io, "$s with $(length(names)) entries:")
    for n in names
        println(io, "  :$n => $(summary(t[n]))")
    end
end

Base.keys(t::AbstractTraces{names}) where {names} = names
Base.haskey(t::AbstractTraces{names}, k::Symbol) where {names} = k in names

#####

"""
Dedicated for `MultiplexTraces` to avoid scalar indexing when `view(view(t::MultiplexTrace, 1:end-1), I)`.
"""
struct RelativeTrace{left,right,T,E} <: AbstractTrace{E}
    trace::Trace{T,E}
end
RelativeTrace{left,right}(t::Trace{T,E}) where {left,right,T,E} =
    RelativeTrace{left,right,T,E}(t)

Base.size(x::RelativeTrace{0,-1}) = (max(0, length(x.trace) - 1),)
Base.size(x::RelativeTrace{1,0}) = (max(0, length(x.trace) - 1),)
Base.getindex(s::RelativeTrace{0,-1}, I) = getindex(s.trace, I)
Base.getindex(s::RelativeTrace{1,0}, I) = getindex(s.trace, I .+ 1)
Base.setindex!(s::RelativeTrace{0,-1}, v, I) = setindex!(s.trace, v, I)
Base.setindex!(s::RelativeTrace{1,0}, v, I) = setindex!(s.trace, v, I .+ 1)
Base.eltype(t::RelativeTrace) = eltype(t.trace)
capacity(t::RelativeTrace) = capacity(t.trace)

"""
    MultiplexTraces{names}(trace)

A special [`AbstractTraces`](@ref) which has exactly two traces of the same
length. And those two traces share the header and tail part.

For example, if a `trace` contains elements between 0 and 9, then the first
`trace_A` is a view of elements from 0 to 8 and the second one is a view from 1
to 9.

```
      ┌─────trace_A───┐
trace 0 1 2 3 4 5 6 7 8 9
        └────trace_B────┘
```

This is quite common in RL to represent `states` and `next_states`.
"""
struct MultiplexTraces{names,T,E} <: AbstractTraces{names,Tuple{E,E}}
    trace::T
end

function MultiplexTraces{names}(t) where {names}
    if length(names) != 2
        throw(
            ArgumentError(
                "MultiplexTraces has exactly two sub traces, got $(length(names)) trace names",
            ),
        )
    end
    trace = convert(AbstractTrace, t)
    MultiplexTraces{names,typeof(trace),eltype(trace)}(trace)
end

Adapt.adapt_structure(to, t::MultiplexTraces{names}) where {names} =
    MultiplexTraces{names}(Adapt.adapt_structure(to, t.trace))

Base.getindex(t::MultiplexTraces{names}, k::Symbol) where {names} = _getindex(t, Val(k))

@generated function _getindex(t::MultiplexTraces{names}, ::Val{k}) where {names,k}
    ex = :()
    if QuoteNode(names[1]) == QuoteNode(k)
        ex = :(RelativeTrace{0,-1}(t.trace))
    elseif QuoteNode(names[2]) == QuoteNode(k)
        ex = :(RelativeTrace{1,0}(t.trace))
    else
        ex = :(throw(ArgumentError("unknown trace name: $k")))
    end
    return :($ex)
end

Base.getindex(t::MultiplexTraces{names}, I::Int) where {names} =
    NamedTuple{names}((t.trace[I], t.trace[I+1]))
Base.getindex(t::MultiplexTraces{names}, I::AbstractArray{Int}) where {names} =
    NamedTuple{names}((t.trace[I], t.trace[I.+1]))

Base.size(t::MultiplexTraces) = (max(0, length(t.trace) - 1),)
capacity(t::MultiplexTraces) = capacity(t.trace)

@forward MultiplexTraces.trace Base.parent, Base.pop!, Base.popfirst!, Base.empty!

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(
        t::MultiplexTraces{names},
        x::NamedTuple{ks,Tuple{Ts}},
    ) where {names,ks,Ts}
        k, v = first(ks), first(x)
        if k in names
            $f(t.trace, v)
        end
    end
    @eval function Base.$f(
        t::MultiplexTraces{names},
        x::RelativeTrace{left,right},
    ) where {names,left,right}
        if left == 0 #do not accept appending the second name as it would be appended twice
            $f(t[first(names)].trace, x.trace)
        end
    end
end

struct Traces{names,T,N,E} <: AbstractTraces{names,E}
    traces::T
end

function Adapt.adapt_structure(to, t::Traces{names,T,N,E}) where {names,T,N,E}
    data = Adapt.adapt_structure(to, t.traces)
    # FIXME: `E` is not adapted here
    Traces{names,typeof(data),length(names),E}(data)
end

function Traces(; kw...)
    data = map(x -> convert(AbstractTrace, x), values(kw))
    names = keys(data)
    Traces{names,typeof(data),length(names),typeof(values(data))}(data)
end


Base.getindex(ts::Traces, s::Symbol) = Base.getindex(ts::Traces, Val(s))

function Base.getindex(ts::Traces, ::Val{s}) where {s}
    t = _gettrace(ts, Val(s))
    if t isa AbstractTrace
        t
    elseif t isa MultiplexTraces
        _getindex(t, Val(s))
    else
        throw(ArgumentError("unknown trace name: $s"))
    end
end

@generated function _gettrace(ts::Traces{names,Trs,N,E}, ::Val{k}) where {names,Trs,N,E,k}
    index_ = build_trace_index(names, Trs)
    # Generate code, i.e. find the correct index for a given key
    ex = :()

    for name in names
        if QuoteNode(name) == QuoteNode(k)
            index_element = index_[k]
            ex = :(ts.traces[$index_element])
            break
        end
    end

    return :($ex)
end

@generated function Base.getindex(t::Traces{names}, i) where {names}
    ex = :(NamedTuple{$(names)}($(Expr(:tuple))))
    for k in names
        push!(ex.args[2].args, :(t[Val($(QuoteNode(k)))][i]))
    end
    return ex
end

function Base.:(+)(t1::AbstractTraces{k1,T1}, t2::AbstractTraces{k2,T2}) where {k1,k2,T1,T2}
    ks = (k1..., k2...)
    ts = (t1, t2)
    Traces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts)
end

function Base.:(+)(t1::AbstractTraces{k1,T1}, t2::Traces{k2,T,N,T2}) where {k1,T1,k2,T,N,T2}
    ks = (k1..., k2...)
    ts = (t1, t2.traces...)
    Traces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts)
end


function Base.:(+)(t1::Traces{k1,T,N,T1}, t2::AbstractTraces{k2,T2}) where {k1,T,N,T1,k2,T2}
    ks = (k1..., k2...)
    ts = (t1.traces..., t2)
    Traces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts)
end

function Base.:(+)(
    t1::Traces{k1,T1,N1,E1},
    t2::Traces{k2,T2,N2,E2},
) where {k1,T1,N1,E1,k2,T2,N2,E2}
    ks = (k1..., k2...)
    ts = (t1.traces..., t2.traces...)
    Traces{ks,typeof(ts),length(ks),Tuple{E1.types...,E2.types...}}(ts)
end

Base.size(t::Traces) = (mapreduce(length, min, t.traces),)
max_length(t::Traces) = mapreduce(length, max, t.traces)

function capacity(t::Traces{names,Trs,N,E}) where {names,Trs,N,E}
    minimum(map(idx -> capacity(t[idx]), names))
end

@generated function Base.push!(ts::Traces, xs::NamedTuple{N,T}) where {N,T}
    ex = :()
    for n in N
        ex = :($ex; push!(ts, Val($(QuoteNode(n))), xs.$n))
    end
    return :($ex)
end

@generated function Base.pushfirst!(ts::Traces, xs::NamedTuple{N,T}) where {N,T}
    ex = :()
    for n in N
        ex = :($ex; pushfirst!(ts, Val($(QuoteNode(n))), xs.$n))
    end
    return :($ex)
end

@generated function Base.pushfirst!(
    ts::Traces{names,Trs,N,E},
    ::Val{k},
    v,
) where {names,Trs,N,E,k}
    index_ = build_trace_index(names, Trs)
    # Generate code, i.e. find the correct index for a given key
    ex = :()

    for name in names
        if QuoteNode(name) == QuoteNode(k)
            index_element = index_[k]
            ex = :(pushfirst!(ts.traces[$index_element], Val($(QuoteNode(k))), v))
            break
        end
    end

    return :($ex)
end

@generated function Base.push!(
    ts::Traces{names,Trs,N,E},
    ::Val{k},
    v,
) where {names,Trs,N,E,k}
    index_ = build_trace_index(names, Trs)
    # Generate code, i.e. find the correct index for a given key
    ex = :()

    for name in names
        if QuoteNode(name) == QuoteNode(k)
            index_element = index_[k]
            ex = :(push!(ts.traces[$index_element], Val($(QuoteNode(k))), v))
            break
        end
    end

    return :($ex)
end

for f in (:push!, :pushfirst!)
    @eval function Base.$f(t::AbstractTrace, ::Val{k}, v) where {k}
        $f(t, v)
    end

    @eval function Base.$f(t::Trace, ::Val{k}, v) where {k}
        $f(t.parent, v)
    end

    @eval function Base.$f(ts::MultiplexTraces, ::Val{k}, v) where {k}
        $f(ts, (; k => v))
    end
end


for f in (:append!, :prepend!)
    @eval function Base.$f(ts::Traces, xs::Traces)
        for k in keys(xs)
            t = _gettrace(ts, Val(k))
            $f(t, xs[k])
        end
    end
end

for f in (:pop!, :popfirst!, :empty!)
    @eval function Base.$f(ts::Traces)
        for t in ts.traces
            $f(t)
        end
    end
end


"""
    build_trace_index(names::NTuple, traces_signature::DataType)

Take type signature from `Traces` and build a mapping from trace name to trace index
"""
function build_trace_index(names::NTuple, traces_signature::DataType)
    # Build index
    index_ = Dict()

    if traces_signature <: NamedTuple
        # Handle simple Traces
        index_ = Dict(name => i for (name, i) ∈ zip(names, 1:length(names)))
    elseif traces_signature <: Tuple
        # Handle MultiplexTracesup
        i = 1
        j = 1
        trace_list = traces_signature.parameters
        for tr in trace_list
            if tr <: MultiplexTraces
                index_[names[i]] = j
                i += 1
                index_[names[i]] = j
            else
                index_[names[i]] = j
            end
            i += 1
            j += 1
        end
    else
        error("Traces store is neither a tuple nor a named tuple!")
    end
    return index_
end
