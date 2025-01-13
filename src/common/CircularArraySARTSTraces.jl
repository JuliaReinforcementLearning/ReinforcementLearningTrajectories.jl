export CircularArraySARTSTraces

import CircularArrayBuffers.CircularArrayBuffer

const CircularArraySARTSTraces = Traces{
    SS′ART,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:CircularArrayBuffer}},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
    }
}

function CircularArraySARTSTraces(;
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
    Traces(
        action = CircularArrayBuffer{action_eltype}(action_size..., capacity),
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

CircularArrayBuffers.capacity(t::CircularArraySARTSTraces) = minimum(map(capacity,t.traces))
