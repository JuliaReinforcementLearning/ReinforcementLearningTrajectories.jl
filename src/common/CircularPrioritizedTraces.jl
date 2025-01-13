export CircularPrioritizedTraces

using CircularArrayBuffers: CircularVectorBuffer

struct CircularPrioritizedTraces{T,names,Ts} <: AbstractTraces{names,Ts}
    keys::CircularVectorBuffer{Int,Vector{Int}}
    priorities::SumTree{Float32}
    traces::T
    default_priority::Float32
end

function CircularPrioritizedTraces(
    traces::AbstractTraces{names,Ts};
    default_priority,
) where {names,Ts}
    new_names = (:key, :priority, names...)
    new_Ts = Tuple{Int,Float32,Ts.parameters...}
    c = capacity(traces)
    CircularPrioritizedTraces{typeof(traces),new_names,new_Ts}(
        CircularVectorBuffer{Int}(c),
        SumTree(c),
        traces,
        default_priority,
    )
end

function Base.push!(t::CircularPrioritizedTraces, x)
    push!(t.traces, x)
    if length(t.traces) == 1
        push!(t.keys, 1)
        push!(t.priorities, t.default_priority)
    elseif length(t.traces) > 1
        push!(t.keys, t.keys[end] + 1)
        push!(t.priorities, t.default_priority)
    else
        # may be partial inserting at the first step, ignore it
    end
end

function Base.setindex!(t::CircularPrioritizedTraces, vs, k::Symbol, keys)
    if k === :priority
        @assert length(vs) == length(keys)
        for (i, v) in zip(keys, vs)
            if t.keys[1] <= i <= t.keys[end]
                t.priorities[i-t.keys[1]+1] = v
            end
        end
    else
        @error "unsupported yet"
    end
end

Base.size(t::CircularPrioritizedTraces) = size(t.traces)
max_length(t::CircularPrioritizedTraces) = max_length(t.traces)

function Base.getindex(ts::CircularPrioritizedTraces, s::Symbol)
    if s === :priority
        Trace(ts.priorities)
    elseif s === :key
        Trace(ts.keys)
    else
        ts.traces[s]
    end
end

Base.getindex(t::CircularPrioritizedTraces{<:Any,names}, i) where {names} =
    NamedTuple{names}(map(k -> t[k][i], names))

capacity(t::CircularPrioritizedTraces) =
    ReinforcementLearningTrajectories.capacity(t.traces)
