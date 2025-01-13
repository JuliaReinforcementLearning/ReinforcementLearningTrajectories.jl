export ElasticArraySLARTTraces

const ElasticArraySLARTTraces = Traces{
    SS′LL′AA′RT,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:ElasticArray}},
        <:MultiplexTraces{LL′,<:Trace{<:ElasticArray}},
        <:MultiplexTraces{AA′,<:Trace{<:ElasticArray}},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
    },
}

function ElasticArraySLARTTraces(;
    capacity::Int,
    state = Int => (),
    legal_actions_mask = Bool => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    legal_actions_mask_eltype, legal_actions_mask_size = legal_actions_mask
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(ElasticArray{state_eltype}(undef, state_size..., 0)) +
    MultiplexTraces{LL′}(
        ElasticArray{legal_actions_mask_eltype}(undef, legal_actions_mask_size..., 0),
    ) +
    MultiplexTraces{AA′}(ElasticArray{action_eltype}(undef, action_size..., 0)) +
    Traces(
        reward = ElasticArray{reward_eltype}(undef, reward_size..., 0),
        terminal = ElasticArray{terminal_eltype}(undef, terminal_size..., 0),
    )
end
