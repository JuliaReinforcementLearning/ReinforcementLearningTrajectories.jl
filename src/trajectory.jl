export Trajectory, TrajectoryStyle, SyncTrajectoryStyle, AsyncTrajectoryStyle

using Base.Threads

struct AsyncTrajectoryStyle end
struct SyncTrajectoryStyle end

"""
    Trajectory(container, sampler, controller)

The `container` is used to store experiences. Common ones are [`Traces`](@ref)
or [`Episodes`](@ref). The `sampler` is used to sample experience batches from
the `container`. The `controller` controls whether it is time to sample a batch
or not.

Supported methoes are:

- `push!(t::Trajectory, experience)`, add one experience into the trajectory.
- `append!(t::Trajectory, batch)`, add a batch of experiences into the trajectory.
- `take!(t::Trajectory)`, take a batch of experiences from the trajectory. Note
  that `nothing` may be returned, indicating that it's not ready to sample yet.
"""
Base.@kwdef struct Trajectory{C,S,T,F}
    container::C
    sampler::S = DummySampler()
    controller::T = InsertSampleRatioController()
    transformer::F = identity

    function Trajectory(
        c::C,
        s::S,
        t::T = InsertSampleRatioController(),
        f = identity,
    ) where {C,S,T}
        if c isa EpisodesBuffer
            new{C,S,T,typeof(f)}(c, s, t, f)
        else
            eb = EpisodesBuffer(c)
            new{typeof(eb),S,T,typeof(f)}(eb, s, t, f)
        end
    end

    function Trajectory(
        container::C,
        sampler::S,
        controller::T,
        transformer,
    ) where {C,S,T<:AsyncInsertSampleRatioController}
        t = Threads.@spawn while true
            for msg in controller.ch_in
                if msg.f === Base.push!
                    x, = msg.args
                    msg.f(container, x)
                    controller.n_inserted += 1
                elseif msg.f === Base.append!
                    x, = msg.args
                    msg.f(container, x)
                    controller.n_inserted += length(x)
                else
                    msg.f(container, msg.args...; msg.kw...)
                end

                if controller.n_inserted >= controller.threshold
                    if controller.n_sampled <=
                       (controller.n_inserted - controller.threshold) * controller.ratio
                        batch = StatsBase.sample(sampler, container)
                        put!(controller.ch_out, batch)
                        controller.n_sampled += 1
                    end
                end
            end
        end

        bind(controller.ch_in, t)
        bind(controller.ch_out, t)

        new{C,S,T,typeof(transformer)}(container, sampler, controller, transformer)
    end
end

TrajectoryStyle(::Trajectory) = SyncTrajectoryStyle()
TrajectoryStyle(::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}) =
    AsyncTrajectoryStyle()

Base.bind(::Trajectory, ::Task) = nothing

function Base.bind(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, task)
    bind(t.controler.ch_in, task)
    bind(t.controler.ch_out, task)
end

Base.setindex!(t::Trajectory, v, I...) = setindex!(t.container, v, I...)

#####
# in
#####

struct CallMsg
    f::Any
    args::Tuple
    kw::Any
end

Base.push!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, x) =
    put!(t.controller.ch_in, CallMsg(Base.push!, (x,), NamedTuple()))
Base.append!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, x) =
    put!(t.controller.ch_in, CallMsg(Base.append!, (x,), NamedTuple()))
Base.setindex!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, v, I...) =
    put!(t.controller.ch_in, CallMsg(Base.setindex!, (v, I...), NamedTuple()))

function Base.append!(t::Trajectory, x)
    append!(t.container, x)
    on_insert!(t.controller, length(x), x)
end

# !!! by default we assume `x`  is a complete example which contains all the traces
# When doing partial inserting, the result of undefined
function Base.push!(t::Trajectory, x)
    push!(t.container, x)
    on_insert!(t, x)
end

function Base.push!(t::Trajectory, x::PartialNamedTuple) #used at EpisodesBuffer
    push!(t.container, x)
    on_insert!(t, x.namedtuple)
end

on_insert!(t::Trajectory, x) = on_insert!(t, 1, x)
on_insert!(t::Trajectory, n::Int, x) = on_insert!(t.controller, n, x)

#####
# out
#####

SampleGenerator(t::Trajectory) = SampleGenerator(t.sampler, t.container) #currently not in use

on_sample!(t::Trajectory) = on_sample!(t.controller)
StatsBase.sample(t::Trajectory) = StatsBase.sample(t.sampler, t.container)

"""
Keep sampling batches from the trajectory until the trajectory is not ready to
be sampled yet due to the `controller`.
"""
iter(t::Trajectory) =
    Iterators.takewhile(_ -> on_sample!(t), Iterators.cycle(SampleGenerator(t)))

#The use of iterate(::SampleGenerator) has been suspended in v0.1.8 due to a significant drop in performance. 
function Base.iterate(t::Trajectory, args...)
    if length(t.container) > 0 && on_sample!(t)
        StatsBase.sample(t), nothing
    else
        nothing
    end
end
Base.IteratorSize(t::Trajectory) = Base.IteratorSize(iter(t))

Base.iterate(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, args...) =
    iterate(t.controller.ch_out, args...)
Base.IteratorSize(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}) =
    Base.IteratorSize(t.controller.ch_out)

Base.keys(t::Trajectory) = keys(t.container)
Base.haskey(t::Trajectory, k) = k in keys(t)
