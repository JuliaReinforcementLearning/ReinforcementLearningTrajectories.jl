import ReinforcementLearningTrajectories.fetch
@testset "Samplers" begin
    @testset "BatchSampler" begin
        sz = 32
        s = BatchSampler(sz)
        t = Traces(state = rand(3, 4, 5), action = rand(1:4, 5))

        b = sample(s, t)

        @test keys(b) == (:state, :action)
        @test size(b.state) == (3, 4, sz)
        @test size(b.action) == (sz,)

        #In EpisodesBuffer
        eb = EpisodesBuffer(CircularArraySARTSATraces(capacity = 10))
        push!(eb, (state = 1,))
        for i = 1:5
            push!(eb, (state = i + 1, action = i, reward = i, terminal = false))
        end
        push!(eb, PartialNamedTuple((next_action = 6,)))
        push!(eb, (state = 7,))
        for (j, i) in enumerate(8:11)
            push!(eb, (state = i, action = i - 1, reward = i - 1, terminal = false))
        end
        push!(eb, PartialNamedTuple((next_action = 11,)))

        s = BatchSampler(1000)
        b = sample(s, eb)
        cm = counter(b[:state])
        @test !haskey(cm, 6)
        @test !haskey(cm, 11)
        @test all(in(keys(cm)), [1:5; 7:10])
    end

    @testset "MetaSampler" begin
        t = Trajectory(
            container = Traces(a = Int[], b = Bool[]),
            sampler = MetaSampler(policy = BatchSampler(3), critic = BatchSampler(5)),
        )
        push!(t, (a = 1,))
        for i = 1:10
            push!(t, (a = i, b = true))
        end

        batches = collect(t)

        @test length(batches) == 11
        @test length(batches[1][:policy][:a]) == 3 && length(batches[1][:critic][:b]) == 5
    end

    @testset "MultiBatchSampler" begin
        t = Trajectory(
            container = Traces(a = Int[], b = Bool[]),
            sampler = MetaSampler(
                policy = BatchSampler(3),
                critic = MultiBatchSampler(BatchSampler(5), 2),
            ),
        )

        push!(t, (a = 1,))
        for i = 1:10
            push!(t, (a = i, b = true))
        end

        batches = collect(t)

        @test length(batches) == 11
        @test length(batches[1][:policy][:a]) == 3
        @test length(batches[1][:critic]) == 2 # we sampled 2 batches for critic
        @test length(batches[1][:critic][1][:b]) == 5 #each batch is 5 samples
    end

    #! format: off
    @testset "NStepSampler" begin
        γ = 0.99
        n_stack = 2
        n_horizon = 3
        batchsize = 1000
        eb = EpisodesBuffer(CircularArraySARTSATraces(capacity=10))
        s1 = NStepBatchSampler(eb, n=n_horizon, γ=γ, stacksize=n_stack, batchsize=batchsize)

        push!(eb, (state = 1,))
        for i = 1:5
            push!(eb, (state = i+1, action =i, reward = i, terminal = i == 5))
        end
        push!(eb, PartialNamedTuple((next_action = 6,)))
        push!(eb, (state = 7,))
        for (j,i) = enumerate(8:11)
            push!(eb, (state = i, action = i-1, reward = i-1, terminal = false))
        end
        push!(eb, PartialNamedTuple((next_action = 11,)))

        weights, ns = ReinforcementLearningTrajectories.valid_range(s1, eb)
        @test weights == [0,1,1,1,1,0,0,1,1,1,0]
        @test ns == [3,3,3,2,1,-1,3,3,2,1,0] #the -1 is due to ep_lengths[6] being that of 2nd episode but step_numbers[6] being that of 1st episode
        inds = [i for i in eachindex(weights) if weights[i] == 1]
        batch = sample(s1, eb)
        for key in keys(eb)
            @test haskey(batch, key)
        end
        #state: samples with stacksize
        states = ReinforcementLearningTrajectories.fetch(s1, eb[:state], Val(:state), inds, ns[inds])
        @test states == [1 2 3 4 7 8 9;
                         2 3 4 5 8 9 10]
        @test all(in(eachcol(states)), unique(eachcol(batch[:state])))
        #next_state: samples with stacksize and nsteps forward
        next_states = ReinforcementLearningTrajectories.fetch(s1, eb[:next_state], Val(:next_state), inds, ns[inds])
        @test next_states == [4 5 5 5 10 10 10;
                              5 6 6 6 11 11 11]
        @test all(in(eachcol(next_states)), unique(eachcol(batch[:next_state])))
        # action: samples normally
        actions = ReinforcementLearningTrajectories.fetch(s1, eb[:action], Val(:action), inds, ns[inds])
        @test actions == inds
        @test all(in(actions), unique(batch[:action]))
        #next_action: is a multiplex trace: should automatically sample nsteps forward
        next_actions = ReinforcementLearningTrajectories.fetch(s1, eb[:next_action], Val(:next_action), inds, ns[inds])
        @test next_actions == [5, 6, 6, 6, 11, 11, 11]
        @test all(in(next_actions), unique(batch[:next_action]))
        #reward: discounted sum
        rewards = ReinforcementLearningTrajectories.fetch(s1, eb[:reward], Val(:reward), inds, ns[inds])
        @test rewards ≈ [2+0.99*3+0.99^2*4, 3+0.99*4+0.99^2*5, 4+0.99*5, 5, 8+0.99*9+0.99^2*10,9+0.99*10, 10]
        @test all(in(rewards), unique(batch[:reward]))
        #terminal: nsteps forward
        terminals = ReinforcementLearningTrajectories.fetch(s1, eb[:terminal], Val(:terminal), inds, ns[inds])
        @test terminals == [0,1,1,1,0,0,0]

        ### CircularPrioritizedTraces and NStepBatchSampler
        γ = 0.99
        n_horizon = 3
        batchsize = 4
        eb = EpisodesBuffer(CircularPrioritizedTraces(CircularArraySARTSATraces(capacity=10), default_priority = 10f0))
        s1 = NStepBatchSampler(eb, n=n_horizon, γ=γ, batchsize=batchsize)

        push!(eb, (state = 1,))
        for i = 1:5
            push!(eb, (state = i+1, action = i, reward = i, terminal = i == 5))
        end
        push!(eb, PartialNamedTuple((action=6,)))
        push!(eb, (state = 7,))
        for (j,i) = enumerate(7:10)
            push!(eb, (state = i+1, action = i, reward = i, terminal = i==10))
        end
        push!(eb, PartialNamedTuple((action = 11,)))
        weights, ns = ReinforcementLearningTrajectories.valid_range(s1, eb)
        inds = [i for i in eachindex(weights) if weights[i] == 1]
        batch = sample(s1, eb)
        for key in (keys(eb)..., :key, :priority)
            @test haskey(batch, key)
        end
    end

    @testset "EpisodesSampler" begin
        s = EpisodesSampler()
        eb = EpisodesBuffer(CircularArraySARTSTraces(capacity=10))
        push!(eb, (state = 1,))
        for i = 1:5
            push!(eb, (state = i+1, action = i, reward = i, terminal = false))
        end
        push!(eb, (state = 7,))
        for (j,i) = enumerate(8:12)
            push!(eb, (state = i, action = i-1, reward = i-1, terminal = false))
        end

        b = sample(s, eb)
        @test length(b) == 2
        @test b[1][:state] == [2:5;]
        @test b[1][:next_state] == [3:6;]
        @test b[1][:action] == [2:5;]
        @test b[1][:reward] == [2:5;]
        @test b[2][:state] == [7:11;]
        @test b[2][:next_state] == [8:12;]
        @test b[2][:action] == [7:11;]
        @test b[2][:reward] == [7:11;]

        for (j,i) = enumerate(2:5)
            push!(eb, (state = i, action =i, reward = i-1, terminal = false))
        end
        #only the last state of the first episode is still buffered. Should not be sampled.
        b = sample(s, eb)
        @test length(b) == 1

        # with specified traces
        s = EpisodesSampler{(:state,)}()
        eb = EpisodesBuffer(CircularArraySARTSTraces(capacity=10))
        push!(eb, (state = 1,))
        for i = 1:5
            push!(eb, (state = i+1, action = i, reward = i, terminal = false))
        end
        push!(eb, (state = 7,))
        for (j,i) = enumerate(8:12)
            push!(eb, (state = i, action = i-1, reward = i-1, terminal = false))
        end

        b = sample(s, eb)
        @test length(b) == 2
        @test length(b[1][:state]) == 4
        @test length(b[2][:state]) == 5
        @test !haskey(b[1], :action)
    end
    @testset "MultiStepSampler" begin
        n_stack = 2
        n_horizon = 3
        batchsize = 1000
        eb = EpisodesBuffer(CircularArraySARTSATraces(capacity=10))
        s1 = MultiStepSampler(eb, n=n_horizon, stacksize=n_stack, batchsize=batchsize)

        push!(eb, (state = 1,))
        for i = 1:5
            push!(eb, (state = i+1, action = i, reward = i, terminal = i == 5))
        end
        push!(eb, PartialNamedTuple((action=6,)))
        push!(eb, (state = 7,))
        for (j,i) = enumerate(8:11)
            push!(eb, (state = i, action = i-1, reward = i-1, terminal = false))
        end
        push!(eb, PartialNamedTuple((action=11,)))

        weights, ns = ReinforcementLearningTrajectories.valid_range(s1, eb)
        @test weights == [0,1,1,1,1,0,0,1,1,1,0]
        @test ns == [3,3,3,2,1,-1,3,3,2,1,0] # the -1 is due to ep_lengths[6] being that of 2nd episode but step_numbers[6] being that of 1st episode
        inds = [i for i in eachindex(weights) if weights[i] == 1]
        batch = sample(s1, eb)
        for key in keys(eb)
            @test haskey(batch, key)
        end

        # state and next_state: samples with stacksize
        states =  ReinforcementLearningTrajectories.fetch(s1, eb[:state], Val(:state), inds, ns[inds])
        @test states == [[1 2 3; 2 3 4], [2 3 4; 3 4 5], [3 4; 4 5], [4; 5;;], [7 8 9; 8 9 10], [8 9; 9 10], [9; 10;;]]
        @test all(in(states), batch[:state])
        # next_state: samples with stacksize and nsteps forward
        next_states = ReinforcementLearningTrajectories.fetch(s1, eb[:next_state], Val(:next_state), inds, ns[inds])
        @test next_states == [[2 3 4; 3 4 5], [3 4 5; 4 5 6], [4 5; 5 6], [5; 6;;], [8 9 10; 9 10 11], [9 10; 10 11], [10; 11;;]]
        @test all(in(next_states), batch[:next_state])
        # all other traces sample normally
        actions = ReinforcementLearningTrajectories.fetch(s1, eb[:action], Val(:action), inds, ns[inds])
        @test actions == [[2,3,4], [3,4,5], [4,5], [5], [8,9,10], [9,10],[10]]
        @test all(in(actions), batch[:action])
        next_actions = ReinforcementLearningTrajectories.fetch(s1, eb[:next_action], Val(:next_action), inds, ns[inds])
        @test next_actions == [a .+ 1 for a in [[2,3,4], [3,4,5], [4,5], [5], [8,9,10], [9,10],[10]]]
        @test all(in(next_actions), batch[:next_action])
        rewards = ReinforcementLearningTrajectories.fetch(s1, eb[:reward], Val(:reward), inds, ns[inds])
        @test rewards == actions
        @test all(in(rewards), batch[:reward])
        terminals = ReinforcementLearningTrajectories.fetch(s1, eb[:terminal], Val(:terminal), inds, ns[inds])
        @test terminals == [[a == 5 ? 1 : 0 for a in acs] for acs in actions]
    end
end
