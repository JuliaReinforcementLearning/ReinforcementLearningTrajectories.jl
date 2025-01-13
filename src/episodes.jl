export EpisodesBuffer, PartialNamedTuple
import DataStructures.CircularBuffer
using ElasticArrays: ElasticArray, ElasticVector

"""
    EpisodesBuffer(traces::AbstractTraces)

Wraps an `AbstractTraces` object, usually the container of a `Trajectory`.
`EpisodesBuffer` tracks the indexes of the `traces` object that belong to the same episodes.
To that end, it stores
1. an vector `sampleable_inds` of Booleans that determine whether an index in Traces is legally sampleable
(i.e., it is not the index of a last state of an episode);
2. a vector `episodes_lengths` that contains the total duration of the episode that each step belong to;
3. an vector `step_numbers` that contains the index within the episode of the corresponding step.

This information is used to correctly sample the traces. For example, if we have an episode that lasted 10 steps, the buffer stores 11 states. sampleable_inds[i] will
be true for the index of the first ten steps, and 0 at the index of state number 11. episodes_lengths will be 11
consecutive 10s (the episode saw 11 states but 10 steps occured). step_numbers will be 1 to 11.

If `traces` is a capacitated buffer, such as a CircularArraySARTSTraces, then these three vectors will also be circular.

EpisodesBuffer assumes that individual transitions are `push!`ed. Appending is not yet supported.
"""

mutable struct EpisodesBuffer{names, E, T<:AbstractTraces{names, E},B,S} <: AbstractTraces{names,E}
    traces::T
    sampleable_inds::S
    step_numbers::B
    episodes_lengths::B
end

"""
    PartialNamedTuple(::NamedTuple)

Wraps a NamedTuple to signal an EpisodesBuffer that it is pushed into that it should
ignore the fact that this is a partial insertion. Used at the end of an episode to
complete multiplex traces before moving to the next episode.
"""
struct PartialNamedTuple{T}
    namedtuple::T
end

# Capacity of an EpisodesBuffer is the capacity of the underlying traces + 1 for certain cases
function is_capacity_plus_one(traces::AbstractTraces)
    if any(t->t isa MultiplexTraces, traces.traces)
        # MultiplexTraces buffer next_state or next_action, so we need to add one to the capacity
        return true
    else
        false
    end
end
is_capacity_plus_one(traces::CircularPrioritizedTraces) = is_capacity_plus_one(traces.traces)

function EpisodesBuffer(traces::AbstractTraces)
    cap = is_capacity_plus_one(traces) ? capacity(traces) + 1 : capacity(traces)
    @assert isempty(traces) "EpisodesBuffer must be initialized with empty traces."
    if !isinf(cap)
        legalinds =  CircularBuffer{Bool}(cap)
        step_numbers = CircularBuffer{Int}(cap)
        eplengths = deepcopy(step_numbers)
        EpisodesBuffer(traces, legalinds, step_numbers, eplengths)
    else
        legalinds =  BitVector()
        step_numbers = Vector{Int}()
        eplengths = deepcopy(step_numbers)
        EpisodesBuffer(traces, legalinds, step_numbers, eplengths)
    end
end

function Base.getindex(es::EpisodesBuffer, idx::Int...)
    @boundscheck all(es.sampleable_inds[idx...]) || throw(BoundsError(es.sampleable_inds, idx))
    getindex(es.traces, idx...)
end

function Base.getindex(es::EpisodesBuffer, idx...)
    getindex(es.traces, idx...)
end

Base.setindex!(eb::EpisodesBuffer, idx...) = setindex!(eb.traces, idx...)
capacity(eb::EpisodesBuffer) = capacity(eb.traces)
Base.size(eb::EpisodesBuffer) = size(eb.traces)
Base.length(eb::EpisodesBuffer) = length(eb.traces)
Base.keys(eb::EpisodesBuffer) = keys(eb.traces)
Base.keys(eb::EpisodesBuffer{<:Any,<:Any,<:CircularPrioritizedTraces}) = keys(eb.traces.traces)
function Base.show(io::IO, m::MIME"text/plain", eb::EpisodesBuffer{names}) where {names}
    s = nameof(typeof(eb))
    t = eb.traces
    println(io, "$s containing")
    show(io, m::MIME"text/plain", t)
end

ispartial_insert(traces::Traces, xs) = length(xs) < length(traces.traces) #this is the number of traces it contains not the number of steps.
ispartial_insert(eb::EpisodesBuffer, xs) = ispartial_insert(eb.traces, xs)
ispartial_insert(traces::CircularPrioritizedTraces, xs) = ispartial_insert(traces.traces, xs)

function pad!(trace::Trace)
    pad!(trace.parent)
    return nothing
end

pad!(vect::ElasticArray{T, Vector{T}}) where {T} = push!(vect, zero(T))
pad!(vect::ElasticVector{T, Vector{T}}) where {T} = push!(vect, zero(T))
pad!(buf::CircularArrayBuffer{T,N,A}) where {T,N,A} = push!(buf, zero(T))
pad!(vect::Vector{T}) where {T} = push!(vect, zero(T))

#push a duplicate of last element as a dummy element for all 'trace' objects, ignores multiplex traces, should never be sampled.
@generated function fill_multiplex(trace_tuple::Traces{names,Trs,N,E}) where {names,Trs,N,E}
    traces_signature = Trs
    ex = :()
    i = 1

    if traces_signature <: NamedTuple
        # Handle 'simple' (non-multiplexed) Traces
        for tr in traces_signature.parameters[1]
            ex = :($ex; pad!(trace_tuple.traces[$i])) # pad everything
            i += 1
        end
    elseif traces_signature <: Tuple
        traces_signature = traces_signature.parameters
        for tr in traces_signature
            if !(tr <: MultiplexTraces)
                #push a duplicate of last element as a dummy element, should never be sampled.
                ex = :($ex; pad!(trace_tuple.traces[$i]))
            end
            i += 1
        end
    else
        error("Traces store is neither a tuple nor a named tuple!")
    end

    return :($ex)
end

fill_multiplex(eb::EpisodesBuffer) = fill_multiplex(eb.traces)

fill_multiplex(eb::EpisodesBuffer{<:Any,<:Any,<:CircularPrioritizedTraces}) = fill_multiplex(eb.traces.traces)

max_length(eb::EpisodesBuffer) = max_length(eb.traces)

function Base.push!(eb::EpisodesBuffer, xs::NamedTuple)
    push!(eb.traces, xs)
    partial = ispartial_insert(eb, xs)
    if length(eb.traces) == 0 && partial #first push should be partial
        push!(eb.step_numbers, 1)
        push!(eb.episodes_lengths, 0)
        push!(eb.sampleable_inds, 0)
    elseif !partial #typical inserting
        if haskey(eb,:next_action) # if trace has next_action
            if eb.step_numbers[end] > 1            # and if there are sufficient steps in the current episode
                eb.sampleable_inds[end-1] = 1      # steps are indexable one step later
            end
        else
            eb.sampleable_inds[end] = 1          # otherwise, previous step is now indexable
        end
        push!(eb.sampleable_inds, 0) #this one is no longer
        ep_length = last(eb.step_numbers)
        push!(eb.episodes_lengths, ep_length)
        startidx = max(1,length(eb.step_numbers) - last(eb.step_numbers))
        eb.episodes_lengths[startidx:end] .= ep_length
        push!(eb.step_numbers, ep_length + 1)
    elseif partial
        fill_multiplex(eb)
        eb.sampleable_inds[end] = 0 #previous step is not indexable because it contains the last state
        push!(eb.sampleable_inds, 0) #this one isn't either
        push!(eb.step_numbers, 1)
        push!(eb.episodes_lengths, 0)
    end
    return nothing
end

function Base.push!(eb::EpisodesBuffer, xs::PartialNamedTuple) #wrap a NamedTuple to push without incrementing the step number.
    push!(eb.traces, xs.namedtuple)
    eb.sampleable_inds[end-1] = 1 #completes the episode trajectory.
end

for f in (:pop!, :popfirst!)
    @eval function Base.$f(eb::EpisodesBuffer)
        $f(eb.episodes_lengths)
        $f(eb.sampleable_inds)
        $f(eb.step_numbers)
        $f(eb.traces)
    end
end

function Base.empty!(eb::EpisodesBuffer)
    empty!(eb.traces)
    empty!(eb.episodes_lengths)
    empty!(eb.sampleable_inds)
    empty!(eb.step_numbers)
end
