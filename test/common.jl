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

@testset "CircularArraySARTSATraces" begin
    t = CircularArraySARTSATraces(;
        capacity=3,
        state=Float32 => (2, 3),
        action=Float32 => (2,),
        reward=Float32 => (),
        terminal=Bool => ()
    ) |> gpu

    @test t isa CircularArraySARTSATraces
    @test ReinforcementLearningTrajectories.capacity(t) == 3
    @test CircularArrayBuffers.capacity(t) == 3

    push!(t, (state=ones(Float32, 2, 3),) |> gpu)
    push!(t, (action=ones(Float32, 2), next_state=ones(Float32, 2, 3) * 2) |> gpu)
    @test length(t) == 0

    push!(t, (reward=1.0f0, terminal=false) |> gpu)
    @test length(t) == 0 # next_action is still missing

    push!(t, (action=ones(Float32, 2) * 2,) |> gpu)
    @test length(t) == 1

    push!(t, (state=ones(Float32, 2, 3) * 3,) |> gpu)
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
    push!(t, (state=ones(Float32, 2, 3) * 4, action=ones(Float32, 2) * 3) |> gpu)

    @test length(t) == 2

    push!(t, (reward=3.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 5, action=ones(Float32, 2) * 4) |> gpu)

    @test length(t) == 3

    push!(t, (reward=4.0f0, terminal=false))
    push!(t, (state=ones(Float32, 2, 3) * 6, action=ones(Float32, 2) * 5) |> gpu)
    push!(t, (reward=5.0f0, terminal=false))

    @test length(t) == 3

    push!(t, (action=ones(Float32, 2) * 6,) |> gpu)
    @test length(t) == 3

    # this will trigger the scalar indexing of CuArray
    CUDA.@allowscalar @test t[1] == (
        state=ones(Float32, 2, 3) * 3,
        next_state=ones(Float32, 2, 3) * 4,
        action=ones(Float32, 2) * 3,
        next_action=ones(Float32, 2) * 4,
        reward=3.0f0,
        terminal=false,
    )
    CUDA.@allowscalar @test t[end] == (
        state=ones(Float32, 2, 3) * 5,
        next_state=ones(Float32, 2, 3) * 6,
        action=ones(Float32, 2) * 5,
        next_action=ones(Float32, 2) * 6,
        reward=5.0f0,
        terminal=false,
    )

    batch = t[1:3]
    @test size(batch.state) == (2, 3, 3)
    @test size(batch.action) == (2, 3)
    @test batch.reward == [3.0, 4.0, 5.0] |> gpu
    @test batch.terminal == Bool[0, 0, 0] |> gpu

end

@testset "ElasticArraySARTSTraces" begin
    t = ElasticArraySARTSTraces(;
        state=Float32 => (2, 3),
        action=Int => (),
        reward=Float32 => (),
        terminal=Bool => ()
    )

    @test t isa ElasticArraySARTSTraces

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
    @test ReinforcementLearningTrajectories.capacity(t) == 3
    @test CircularArrayBuffers.capacity(t) == 3
end

@testset "CircularPrioritizedTraces-SARTS" begin
    t = CircularPrioritizedTraces(
        CircularArraySARTSTraces(;
            capacity=3
        ),
        default_priority=1.0f0
    )
    @test ReinforcementLearningTrajectories.capacity(t) == 3

    push!(t, (state=0, action=0))

    for i in 1:5
        push!(t, (reward=1.0f0, terminal=false, state=i, action=i))
    end

    @test length(t) == 3

    s = BatchSampler(5)

    b = sample(s, t)

    t[:priority, [1, 2]] = [0, 0]

    # shouldn't be changed since [1,2] are old keys
    @test t[:priority] == [1.0f0, 1.0f0, 1.0f0]

    t[:priority, [3, 4, 5]] = [0, 1, 0]

    b = sample(s, t)

    @test b.key == [4, 4, 4, 4, 4] # the priority of the rest transitions are set to 0

    #EpisodesBuffer
    t = CircularPrioritizedTraces(
        CircularArraySARTSTraces(;
            capacity=10
        ),
        default_priority=1.0f0
    )

    eb = EpisodesBuffer(t)
    push!(eb, (state = 1, action = 1))
    for i = 1:5
        push!(eb, (state = i+1, action = i+1, reward = i, terminal = false))
    end
    push!(eb, (state = 7, action = 7))
    for (j,i) = enumerate(8:11)
        push!(eb, (state = i, action = i, reward = i-1, terminal = false))
    end
    s = BatchSampler(1000)
    b = sample(s, eb)
    cm = counter(b[:state])
    @test !haskey(cm, 6)
    @test !haskey(cm, 11)
    @test all(in(keys(cm)), [1:5;7:10])


    eb[:priority, [1, 2]] = [0, 0]
    @test eb[:priority] == [zeros(2);ones(8)]
end

@testset "CircularPrioritizedTraces-SARTSA" begin
    t = CircularPrioritizedTraces(
        CircularArraySARTSATraces(;
            capacity=3
        ),
        default_priority=1.0f0
    )
    @test ReinforcementLearningTrajectories.capacity(t) == 3

    push!(t, (state=0, action=0))

    for i in 1:5
        push!(t, (reward=1.0f0, terminal=false, state=i, action=i))
    end

    @test length(t) == 3

    s = BatchSampler(5)

    b = sample(s, t)

    @test t[:priority] == [1.0f0, 1.0f0, 1.0f0]

    t[:priority, [1, 2]] = [0, 0]

    # shouldn't be changed since [1,2] are old keys
    @test t[:priority] == [1.0f0, 1.0f0, 1.0f0]

    t[:priority, [3, 4, 5]] = [0, 1, 0]

    b = sample(s, t)

    @test b.key == [4, 4, 4, 4, 4] # the priority of the rest transitions are set to 0

    #EpisodesBuffer
    t = CircularPrioritizedTraces(
        CircularArraySARTSATraces(;
            capacity=10
        ),
        default_priority=1.0f0
    )

    eb = EpisodesBuffer(t)
    push!(eb, (state = 1,))
    for i = 1:5
        push!(eb, (state = i+1, action = i, reward = i, terminal = false))
    end
    push!(eb, PartialNamedTuple((action = 6,)))
    push!(eb, (state = 7,))
    for i = 8:11
        push!(eb, (state = i, action = i-1, reward = i-1, terminal = false))
    end
    push!(eb, PartialNamedTuple((action=11,)))

    s = BatchSampler(1000)
    b = sample(s, eb)
    cm = counter(b[:state])
    @test !haskey(cm, 6)
    @test !haskey(cm, 11)
    @test all(in(keys(cm)), [1:5;7:10])
end
