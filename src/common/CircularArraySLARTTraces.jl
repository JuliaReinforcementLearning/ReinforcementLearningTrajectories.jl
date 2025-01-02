export CircularArraySLARTTraces

const CircularArraySLARTTraces = Traces{
    SS′LL′AA′RT,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:CircularArrayBuffer}},
        <:MultiplexTraces{LL′,<:Trace{<:CircularArrayBuffer}},
        <:MultiplexTraces{AA′,<:Trace{<:CircularArrayBuffer}},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
    }
}

function CircularArraySLARTTraces(;
    capacity::Int,
    state=Int => (),
    legal_actions_mask=Bool => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    legal_actions_mask_eltype, legal_actions_mask_size = legal_actions_mask
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
    MultiplexTraces{LL′}(CircularArrayBuffer{legal_actions_mask_eltype}(legal_actions_mask_size..., capacity + 1)) +
    MultiplexTraces{AA′}(CircularArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
    Traces(
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

CircularArrayBuffers.capacity(t::CircularArraySLARTTraces) = minimum(map(capacity,t.traces))