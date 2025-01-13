using Random
export EpisodesSampler, Episode, BatchSampler, NStepBatchSampler, MetaSampler, MultiBatchSampler, DummySampler, MultiStepSampler

struct SampleGenerator{S,T}
    sampler::S
    traces::T
end

Base.iterate(s::SampleGenerator) = StatsBase.sample(s.sampler, s.traces), nothing
Base.iterate(s::SampleGenerator, ::Nothing) = nothing

#####
# DummySampler
#####

export DummySampler

"""
Just return the underlying traces.
"""
struct DummySampler end

StatsBase.sample(::DummySampler, t) = t

#####
# BatchSampler
#####

export BatchSampler

struct BatchSampler{names}
    batchsize::Int
    rng::Random.AbstractRNG
end

"""
    BatchSampler{names}(;batchsize, rng=Random.GLOBAL_RNG)
    BatchSampler{names}(batchsize ;rng=Random.GLOBAL_RNG)

Uniformly sample **ONE** batch of `batchsize` examples for each trace specified
in `names`. If `names` is not set, all the traces will be sampled.
"""
BatchSampler(batchsize; kw...) = BatchSampler(; batchsize=batchsize, kw...)
BatchSampler(; kw...) = BatchSampler{nothing}(; kw...)
BatchSampler{names}(batchsize; kw...) where {names} = BatchSampler{names}(; batchsize=batchsize, kw...)
BatchSampler{names}(; batchsize, rng=Random.GLOBAL_RNG) where {names} = BatchSampler{names}(batchsize, rng)

StatsBase.sample(s::BatchSampler{nothing}, t::AbstractTraces) = StatsBase.sample(s, t, keys(t))
StatsBase.sample(s::BatchSampler{names}, t::AbstractTraces) where {names} = StatsBase.sample(s, t, names)

function StatsBase.sample(s::BatchSampler, t::AbstractTraces, names, weights = StatsBase.UnitWeights{Int}(length(t)))
    inds = StatsBase.sample(s.rng, 1:length(t), weights, s.batchsize)
    NamedTuple{names}(map(x -> collect(t[Val(x)][inds]), names))
end

function StatsBase.sample(s::BatchSampler, t::EpisodesBuffer, names)
    StatsBase.sample(s, t.traces, names, StatsBase.FrequencyWeights(t.sampleable_inds[1:end-1]))
end

# !!! avoid iterating an empty trajectory
function Base.iterate(s::SampleGenerator{<:BatchSampler})
    if length(s.traces) > 0
        StatsBase.sample(s.sampler, s.traces), nothing
    else
        nothing
    end
end

#####

StatsBase.sample(s::BatchSampler{nothing}, t::CircularPrioritizedTraces) = StatsBase.sample(s, t, keys(t.traces))

function StatsBase.sample(s::BatchSampler, e::EpisodesBuffer{<:Any, <:Any, <:CircularPrioritizedTraces}, names)
    t = e.traces
    p = collect(deepcopy(t.priorities))
    w = StatsBase.FrequencyWeights(p)
    w .*= e.sampleable_inds[1:length(t)]
    inds = StatsBase.sample(s.rng, eachindex(w), w, s.batchsize)
    NamedTuple{(:key, :priority, names...)}((t.keys[inds], p[inds], map(x -> collect(t.traces[Val(x)][inds]), names)...))
end

function StatsBase.sample(s::BatchSampler, t::CircularPrioritizedTraces, names)
    inds, priorities = rand(s.rng, t.priorities, s.batchsize)
    NamedTuple{(:key, :priority, names...)}((t.keys[inds], priorities, map(x -> collect(t.traces[Val(x)][inds]), names)...))
end

#####
# MetaSampler
#####

export MetaSampler

"""
    MetaSampler(::NamedTuple)

Wraps a NamedTuple containing multiple samplers. When sampled, returns a named tuple with a
batch from each sampler.
Used internally for algorithms that sample multiple times per epoch.
Note that a single "sampling" with a MetaSampler only increases the Trajectory controler
count by 1, not by the number of internal samplers. This should be taken into account when
initializing an agent.


# Example
```
MetaSampler(policy = BatchSampler(10), critic = BatchSampler(100))
```
"""
struct MetaSampler{names,T}
    samplers::NamedTuple{names,T}
end

MetaSampler(; kw...) = MetaSampler(NamedTuple(kw))

StatsBase.sample(s::MetaSampler, t) = map(x -> StatsBase.sample(x, t), s.samplers)

function Base.iterate(s::SampleGenerator{<:MetaSampler})
    if length(s.traces) > 0
        StatsBase.sample(s.sampler, s.traces), nothing
    else
        nothing
    end
end

#####
# MultiBatchSampler
#####

export MultiBatchSampler

"""
    MultiBatchSampler(sampler, n)

Wraps a sampler. When sampled, will sample n batches using sampler. Useful in combination
with MetaSampler to allow different sampling rates between samplers.
Note that a single "sampling" with a MultiBatchSampler only increases the Trajectory
controler count by 1, not by `n`. This should be taken into account when
initializing an agent.

# Example
```
MetaSampler(policy = MultiBatchSampler(BatchSampler(10), 3),
            critic = MultiBatchSampler(BatchSampler(100), 5))
```
"""
struct MultiBatchSampler{S}
    sampler::S
    n::Int
end

StatsBase.sample(m::MultiBatchSampler, t) = [StatsBase.sample(m.sampler, t) for _ in 1:m.n]

function Base.iterate(s::SampleGenerator{<:MultiBatchSampler})
    if length(s.traces) > 0
        StatsBase.sample(s.sampler, s.traces), nothing
    else
        nothing
    end
end

#####
# NStepBatchSampler
#####

export NStepBatchSampler

"""

    NStepBatchSampler{names}(; n, γ, batchsize=32, stacksize=nothing, rng=Random.GLOBAL_RNG)

Used to sample a discounted sum of consecutive rewards in the framework of n-step TD learning.
The "next" element of Multiplexed traces (such as the next_state or the next_action) will be
that in up to `n > 1` steps later in the buffer. The reward will be
the discounted sum of the `n` rewards, with `γ` as the discount factor.

NStepBatchSampler may also be used with n ≥ 1 to sample a "stack" of states if `stacksize` is set
to an integer > 1. This samples the (stacksize - 1) previous states. This is useful in the case
of partial observability, for example when the state is approximated by `stacksize` consecutive
frames.
"""
mutable struct NStepBatchSampler{names, S <: Union{Nothing,Int}, R <: AbstractRNG}
    n::Int # !!! n starts from 1
    γ::Float32
    batchsize::Int
    stacksize::S
    rng::R
end

NStepBatchSampler(t::AbstractTraces; kw...) = NStepBatchSampler{keys(t)}(; kw...)
function NStepBatchSampler{names}(; n, γ, batchsize=32, stacksize=nothing, rng=Random.default_rng()) where {names}
    @assert n >= 1 "n must be ≥ 1."
    ss = stacksize == 1 ? nothing : stacksize
    NStepBatchSampler{names, typeof(ss), typeof(rng)}(n, γ, batchsize, ss, rng)
end

#return a boolean vector of the valid sample indices given the stacksize and the truncated n for each index.
function valid_range(s::NStepBatchSampler, eb::EpisodesBuffer)
    range = copy(eb.sampleable_inds)
    ns = Vector{Int}(undef, length(eb.sampleable_inds))
    stacksize = isnothing(s.stacksize) ? 1 : s.stacksize
    for idx in eachindex(range)
        step_number = eb.step_numbers[idx]
        range[idx] = step_number >= stacksize && eb.sampleable_inds[idx]
        ns[idx] = min(s.n, eb.episodes_lengths[idx] - step_number + 1)
    end
    return range, ns
end

function StatsBase.sample(s::NStepBatchSampler{names}, ts) where {names}
    StatsBase.sample(s, ts, Val(names))
end

function StatsBase.sample(s::NStepBatchSampler, t::EpisodesBuffer, ::Val{names}) where names
    weights, ns = valid_range(s, t)
    inds = StatsBase.sample(s.rng, 1:length(t), StatsBase.FrequencyWeights(weights[1:end-1]), s.batchsize)
    fetch(s, t, Val(names), inds, ns)
end

function fetch(s::NStepBatchSampler, ts::EpisodesBuffer, ::Val{names}, inds, ns) where names
    NamedTuple{names}(map(name -> collect(fetch(s, ts[name], Val(name), inds, ns[inds])), names))
end

#state and next_state have specialized fetch methods due to stacksize
fetch(::NStepBatchSampler{names, Nothing}, trace::AbstractTrace, ::Val{:state}, inds, ns) where {names} = trace[inds]
fetch(s::NStepBatchSampler{names, Int}, trace::AbstractTrace, ::Val{:state}, inds, ns) where {names} = trace[[x + i for i in -s.stacksize+1:0, x in inds]]
fetch(::NStepBatchSampler{names, Nothing}, trace::RelativeTrace{1,0}, ::Val{:next_state}, inds, ns) where {names} = trace[inds .+ ns .- 1]
fetch(s::NStepBatchSampler{names, Int}, trace::RelativeTrace{1,0}, ::Val{:next_state}, inds, ns) where {names}  = trace[[x + ns[idx] - 1 + i for i in -s.stacksize+1:0, (idx,x) in enumerate(inds)]]

#reward due to discounting
function fetch(s::NStepBatchSampler, trace::AbstractTrace, ::Val{:reward}, inds, ns)
    rewards = Vector{eltype(trace)}(undef, length(inds))
    for (i,idx) in enumerate(inds)
        rewards_to_go = trace[idx:idx+ns[i]-1]
        rewards[i] = foldr((x,y)->x + s.γ*y, rewards_to_go)
    end
    return rewards
end
#terminal is that of the nth step
fetch(::NStepBatchSampler, trace::AbstractTrace, ::Val{:terminal}, inds, ns) = trace[inds .+ ns .- 1]
#right multiplex traces must be n-step sampled
fetch(::NStepBatchSampler, trace::RelativeTrace{1,0} , ::Val, inds, ns) = trace[inds .+ ns .- 1]
#normal trace types are fetched at inds
fetch(::NStepBatchSampler, trace::AbstractTrace, ::Val, inds, ns) = trace[inds] #other types of trace are sampled normally

function StatsBase.sample(s::NStepBatchSampler{names}, e::EpisodesBuffer{<:Any, <:Any, <:CircularPrioritizedTraces}) where {names}
    t = e.traces
    p = collect(deepcopy(t.priorities))
    w = StatsBase.FrequencyWeights(p)
    valids, ns = valid_range(s,e)
    w .*= valids[1:length(t)]
    inds = StatsBase.sample(s.rng, eachindex(w), w, s.batchsize)
    merge(
        (key=t.keys[inds], priority=p[inds]),
        fetch(s, e, Val(names), inds, ns)
    )
end

"""
    EpisodesSampler()

A sampler that samples all Episodes present in the Trajectory and divides them into
Episode containers. Truncated Episodes (e.g. due to the buffer capacity) are sampled as well.
There will be at most one truncated episode and it will always be the first one.
"""
struct EpisodesSampler{names}
end

EpisodesSampler() = EpisodesSampler{nothing}()
#EpisodesSampler{names}() = new{names}()


struct Episode{names, N <: NamedTuple{names}}
    nt::N
end

@forward Episode.nt Base.keys, Base.haskey, Base.getindex

StatsBase.sample(s::EpisodesSampler{nothing}, t::EpisodesBuffer) = StatsBase.sample(s,t,keys(t))
StatsBase.sample(s::EpisodesSampler{names}, t::EpisodesBuffer) where names = StatsBase.sample(s,t,names)

function make_episode(t::EpisodesBuffer, range, names)
    nt = NamedTuple{names}(map(x -> collect(t[Val(x)][range]), names))
    Episode(nt)
end

function StatsBase.sample(::EpisodesSampler, t::EpisodesBuffer, names)
    ranges = UnitRange{Int}[]
    idx = 1
    while idx < length(t)
        if t.sampleable_inds[idx] == 1
            last_state_idx = idx + t.episodes_lengths[idx] - t.step_numbers[idx]
            push!(ranges,idx:last_state_idx)
            idx = last_state_idx + 1
        else
            idx += 1
        end
    end

    return [make_episode(t, r, names) for r in ranges]
end

#####MultiStepSampler

"""
    MultiStepSampler{names}(batchsize, n, stacksize, rng)

Sampler that fetches steps `[x, x+1, ..., x + n -1]` for each trace of each sampled index
`x`. The samples are returned in an array of batchsize elements. For each element, n is
truncated by the end of its episode. This means that the dimensions of each sample are not
the same.
"""
struct MultiStepSampler{names, S <: Union{Nothing,Int}, R <: AbstractRNG}
    n::Int
    batchsize::Int
    stacksize::S
    rng::R
end

MultiStepSampler(t::AbstractTraces; kw...) = MultiStepSampler{keys(t)}(; kw...)
function MultiStepSampler{names}(; n::Int, batchsize, stacksize=nothing, rng=Random.default_rng()) where {names}
    @assert n >= 1 "n must be ≥ 1."
    ss = stacksize == 1 ? nothing : stacksize
    MultiStepSampler{names, typeof(ss), typeof(rng)}(n, batchsize, ss, rng)
end

function valid_range(s::MultiStepSampler, eb::EpisodesBuffer)
    range = copy(eb.sampleable_inds)
    ns = Vector{Int}(undef, length(eb.sampleable_inds))
    stacksize = isnothing(s.stacksize) ? 1 : s.stacksize
    for idx in eachindex(range)
        step_number = eb.step_numbers[idx]
        range[idx] = step_number >= stacksize && eb.sampleable_inds[idx]
        ns[idx] = min(s.n, eb.episodes_lengths[idx] - step_number + 1)
    end
    return range, ns
end

function StatsBase.sample(s::MultiStepSampler{names}, ts) where {names}
    StatsBase.sample(s, ts, Val(names))
end

function StatsBase.sample(s::MultiStepSampler, t::EpisodesBuffer, ::Val{names}) where names
    weights, ns = valid_range(s, t)
    inds = StatsBase.sample(s.rng, 1:length(t), StatsBase.FrequencyWeights(weights[1:end-1]), s.batchsize)
    fetch(s, t, Val(names), inds, ns)
end

function fetch(s::MultiStepSampler, ts::EpisodesBuffer, ::Val{names}, inds, ns) where names
    NamedTuple{names}(map(name -> collect(fetch(s, ts[name], Val(name), inds, ns[inds])), names))
end

function fetch(::MultiStepSampler, trace, ::Val, inds, ns)
    [trace[idx:(idx + ns[i] - 1)] for (i,idx) in enumerate(inds)]
end

function fetch(s::MultiStepSampler{names, Int}, trace::AbstractTrace, ::Union{Val{:state}, Val{:next_state}}, inds, ns) where {names}
    [trace[[idx + i + n - 1 for i in -s.stacksize+1:0, n in 1:ns[j]]] for (j,idx) in enumerate(inds)]
end

function StatsBase.sample(s::MultiStepSampler{names}, e::EpisodesBuffer{<:Any, <:Any, <:CircularPrioritizedTraces}) where {names}
    t = e.traces
    p = collect(deepcopy(t.priorities))
    w = StatsBase.FrequencyWeights(p)
    valids, ns = valid_range(s,e)
    w .*= valids[1:length(t)]
    inds = StatsBase.sample(s.rng, eachindex(w), w, s.batchsize)
    merge(
        (key=t.keys[inds], priority=p[inds]),
        fetch(s, e, Val(names), inds, ns)
    )
end
