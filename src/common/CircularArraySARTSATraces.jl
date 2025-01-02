export CircularArraySARTSATraces

import CircularArrayBuffers.CircularArrayBuffer

const CircularArraySARTSATraces = Traces{
    SS′AA′RT,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:CircularArrayBuffer}},
        <:MultiplexTraces{AA′,<:Trace{<:CircularArrayBuffer}},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
    }
}

function CircularArraySARTSATraces(;
    capacity::Int,
    state=Int => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(CircularArrayBuffer{state_eltype}(state_size..., capacity+1)) +
    MultiplexTraces{AA′}(CircularArrayBuffer{action_eltype}(action_size..., capacity+1)) +
    Traces(
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

CircularArrayBuffers.capacity(t::CircularArraySARTSATraces) = minimum(map(capacity,t.traces))
