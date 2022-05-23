@testset "CircularArraySARTTraces" begin
    t = CircularArraySARTTraces(;
        capacity=3,
        state=Float32 => (2, 3),
        action=Float32 => (2),
        reward=Float32 => (),
        terminal=Bool => ()
    )

    push!(t, (state=ones(Float32, 2, 3), action=ones(Float32, 2)))
    @test length(t) == 0

    push!(t, (reward=1.0f0, terminal=false))
    @test length(t) == 0 # next_state and next_action is still missing

    push!(t, (next_state=ones(Float32, 2, 3) * 2, next_action=ones(Float32, 2) * 2))
    @test length(t) == 1

    @test t[1] == (
        state=ones(Float32, 2, 3),
        next_state=ones(Float32, 2, 3) * 2,
        action=ones(Float32, 2),
        next_action=ones(Float32, 2) * 2,
        reward=1.0f0,
        terminal=false,
    )

    push!(t, (reward=2.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 3, action=ones(Float32, 2) * 3))

    @test length(t) == 2

    push!(t, (reward=3.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 4, action=ones(Float32, 2) * 4))

    @test length(t) == 3

    push!(t, (reward=4.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 5, action=ones(Float32, 2) * 5))

    @test length(t) == 3
    @test t[1] == (
        state=ones(Float32, 2, 3) * 2,
        next_state=ones(Float32, 2, 3) * 3,
        action=ones(Float32, 2) * 2,
        next_action=ones(Float32, 2) * 3,
        reward=2.0f0,
        terminal=false,
    )
    @test t[end] == (
        state=ones(Float32, 2, 3) * 4,
        next_state=ones(Float32, 2, 3) * 5,
        action=ones(Float32, 2) * 4,
        next_action=ones(Float32, 2) * 5,
        reward=4.0f0,
        terminal=false,
    )

    batch = t[1:3]
    @test size(batch.state) == (2, 3, 3)
    @test size(batch.action) == (2, 3)
    @test batch.reward == [2.0, 3.0, 4.0]
    @test batch.terminal == Bool[0, 0, 0]
end