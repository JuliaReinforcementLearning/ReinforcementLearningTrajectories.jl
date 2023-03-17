@testset "sum_tree" begin
    t = SumTree(8)

    for i in 1:4
        push!(t, i)
    end

    @test length(t) == 4
    @test size(t) == (4,)

    for i in 5:16
        push!(t, i)
    end

    @test length(t) == 8
    @test size(t) == (8,)
    @test t == 9:16

    t[:] .= 1
    @test t == ones(8)
    @test all([get(t, v)[1] == i for (i, v) in enumerate(0.5:1.0:8)])

    empty!(t)
    @test length(t) == 0
end

@testset "CircularArraySARTTraces" begin
    t = CircularArraySARTTraces(;
        capacity=3,
        state=Float32 => (2, 3),
        action=Float32 => (2,),
        reward=Float32 => (),
        terminal=Bool => ()
    ) |> gpu

    @test t isa CircularArraySARTTraces

    push!(t, (state=ones(Float32, 2, 3), action=ones(Float32, 2)) |> gpu)
    @test length(t) == 0

    push!(t, (reward=1.0f0, terminal=false) |> gpu)
    @test length(t) == 0 # next_state and next_action is still missing

    push!(t, (next_state=ones(Float32, 2, 3) * 2, next_action=ones(Float32, 2) * 2) |> gpu)
    @test length(t) == 1

    # this will trigger the scalar indexing of CuArray
    CUDA.@allowscalar @test t[1] == (
        state=ones(Float32, 2, 3),
        next_state=ones(Float32, 2, 3) * 2,
        action=ones(Float32, 2),
        next_action=ones(Float32, 2) * 2,
        reward=1.0f0,
        terminal=false,
    )

    push!(t, (reward=2.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 3, action=ones(Float32, 2) * 3) |> gpu)

    @test length(t) == 2

    push!(t, (reward=3.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 4, action=ones(Float32, 2) * 4) |> gpu)

    @test length(t) == 3

    push!(t, (reward=4.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 5, action=ones(Float32, 2) * 5) |> gpu)

    @test length(t) == 3

    # this will trigger the scalar indexing of CuArray
    CUDA.@allowscalar @test t[1] == (
        state=ones(Float32, 2, 3) * 2,
        next_state=ones(Float32, 2, 3) * 3,
        action=ones(Float32, 2) * 2,
        next_action=ones(Float32, 2) * 3,
        reward=2.0f0,
        terminal=false,
    )
    CUDA.@allowscalar @test t[end] == (
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
    @test batch.reward == [2.0, 3.0, 4.0] |> gpu
    @test batch.terminal == Bool[0, 0, 0] |> gpu
end

@testset "CircularArraySARTTraces handle missingness" begin
    t = CircularArraySARTTraces(;
        capacity=3,
        state=Float32 => (2, 3),
        action=Float32 => (2,),
        reward=Float32 => (),
        terminal=Bool => ()
    )
    push!(t, (; state=1, action=1, reward=missing, terminal=missing))
    @test length(t) == 0
    push!(t, (; state=1, action=1, reward=1.0, terminal=false))
    @test length(t) == 1
end



@testset "ElasticArraySARTTraces" begin
    t = ElasticArraySARTTraces(;
        state=Float32 => (2, 3),
        action=Int => (),
        reward=Float32 => (),
        terminal=Bool => ()
    )

    @test t isa ElasticArraySARTTraces

    push!(t, (state=ones(Float32, 2, 3), action=1))
    push!(t, (reward=1.0f0, terminal=false, state=ones(Float32, 2, 3) * 2, action=2))

    @test length(t) == 1

    empty!(t)

    @test length(t) == 0
end

@testset "CircularArraySLARTTraces" begin
    t = CircularArraySLARTTraces(;
        capacity=3,
        state=Float32 => (2, 3),
        legal_actions_mask=Bool => (5,),
        action=Int => (),
        reward=Float32 => (),
        terminal=Bool => ()
    )

    @test t isa CircularArraySLARTTraces
end

@testset "CircularPrioritizedTraces" begin
    t = CircularPrioritizedTraces(
        CircularArraySARTTraces(;
            capacity=3
        ),
        default_priority=1.0f0
    )

    push!(t, (state=0, action=0))

    for i in 1:5
        push!(t, (reward=1.0f0, terminal=false, state=i, action=i))
    end

    @test length(t) == 3

    s = BatchSampler(5)

    b = ReinforcementLearningTrajectories.sample(s, t)

    t[:priority, [1, 2]] = [0, 0]

    # shouldn't be changed since [1,2] are old keys
    @test t[:priority] == [1.0f0, 1.0f0, 1.0f0]

    t[:priority, [3, 4, 5]] = [0, 1, 0]

    b = ReinforcementLearningTrajectories.sample(s, t)

    @test b.key == [4, 4, 4, 4, 4] # the priority of the rest transitions are set to 0
end
