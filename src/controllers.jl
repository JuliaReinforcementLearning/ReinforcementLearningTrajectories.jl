export InsertSampleRatioController,
    AsyncInsertSampleRatioController, EpisodeSampleRatioController

"""
    InsertSampleRatioController(;ratio=1., threshold=1)

Used in [`Trajectory`](@ref). The `threshold` means the minimal number of
insertings before sampling. The `ratio` balances the number of insertings and
the number of samplings. For example a ratio of 1/10 will sample once every 10 
insertings in the trajectory.
"""
Base.@kwdef mutable struct InsertSampleRatioController
    ratio::Float64 = 1.0
    threshold::Int = 1
    n_inserted::Int = 0
    n_sampled::Int = 0
end

function on_insert!(c::InsertSampleRatioController, n::Int, ::Any)
    if n > 0
        c.n_inserted += n
    end
end

function on_sample!(c::InsertSampleRatioController)
    if c.n_inserted >= c.threshold && c.n_sampled <= (c.n_inserted - c.threshold) * c.ratio
        c.n_sampled += 1
        return true
    end
    return false
end

#####

mutable struct AsyncInsertSampleRatioController
    ratio::Float64
    threshold::Int
    n_inserted::Int
    n_sampled::Int
    ch_in::Channel
    ch_out::Channel
end

function AsyncInsertSampleRatioController(
    ratio,
    threshold,
    ;
    ch_in_sz = 1,
    ch_out_sz = 1,
    n_inserted = 0,
    n_sampled = 0,
)
    AsyncInsertSampleRatioController(
        ratio,
        threshold,
        n_inserted,
        n_sampled,
        Channel(ch_in_sz),
        Channel(ch_out_sz),
    )
end

"""
    EpisodeSampleRatioController(;ratio=1., threshold=1)

Used in [`Trajectory`](@ref). The `threshold` means the minimal number of
episodes completed before sampling is allowed. The `ratio` balances the number of episodes and
the number of samplings. For example a ratio of 1/10 will sample once every 10 
episodes in the trajectory. Currently only works for environemnts with terminal states. 
"""
Base.@kwdef mutable struct EpisodeSampleRatioController
    ratio::Float64 = 1.0
    threshold::Int = 1
    n_episodes::Int = 0
    n_sampled::Int = 0
end

function on_insert!(c::EpisodeSampleRatioController, n::Int, x::NamedTuple)
    if n > 0
        c.n_episodes += sum(x.terminal)
    end
end

function on_sample!(c::EpisodeSampleRatioController)
    if c.n_episodes >= c.threshold && c.n_sampled <= (c.n_episodes - c.threshold) * c.ratio
        c.n_sampled += 1
        return true
    end
    return false
end
