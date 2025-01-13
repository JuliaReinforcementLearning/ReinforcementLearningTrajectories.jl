using ReinforcementLearningTrajectories
using CircularArrayBuffers
using Test

@testset "EpisodesBuffer" begin
    @testset "with circular SARTS traces" begin
        eb = EpisodesBuffer(CircularArraySARTSTraces(; capacity = 10))

        # push first episode (five steps)
        push!(eb, (state = 1,))
        @test eb.sampleable_inds[end] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        for i = 1:5
            push!(eb, (state = i + 1, action = i, reward = i, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(i, i + 1)
        end
        @test eb[end] ==
              (state = 5, next_state = 6, action = 5, reward = 5, terminal = false)
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0]
        @test length(eb.traces) == 5

        # start second episode
        push!(eb, (state = 7,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 0]
        @test eb[:reward][6] == 0 # 6 is not a valid index, filled with dummy value zero
        @test_throws BoundsError eb[6] # 6 is not a valid index
        @test_throws BoundsError eb[7] # 7 is not a valid index

        # push four steps of second episode
        ep2_len = 0
        for (i, s) in enumerate(8:11)
            ep2_len += 1
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb[end] ==
              (state = 10, next_state = 11, action = 10, reward = 10, terminal = false)
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        @test length(eb) == 10
        # push two more steps of second episode, which replace the oldest steps in the buffer
        for (i, s) in enumerate(12:13)
            ep2_len += 1
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1 + 4
            @test eb.episodes_lengths[end-ep2_len:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb[end] ==
              (state = 12, next_state = 13, action = 12, reward = 12, terminal = false)
        @test eb.sampleable_inds == [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]

        # verify episode 2
        for (i, s) in enumerate(3:13)
            if i in (4, 11)
                @test eb.sampleable_inds[i] == 0
                continue
            else
                @test eb.sampleable_inds[i] == 1
            end
            b = eb[i]
            @test b[:state] == b[:action] == b[:reward] == s
            @test b[:next_state] == s + 1
        end

        # push third episode
        push!(eb, (state = 14,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        #push until it reaches it own start
        for (i, s) in enumerate(15:26)
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
        end
        @test eb.sampleable_inds == [fill(true, 10); [false]]
        @test eb.episodes_lengths == fill(length(15:26), 11)
        @test eb.step_numbers == [3:13;]
        popfirst!(eb)
        @test length(eb) ==
              length(eb.sampleable_inds) - 1 ==
              length(eb.step_numbers) - 1 ==
              length(eb.episodes_lengths) - 1 ==
              9
        @test first(eb.step_numbers) == 4
        pop!(eb)
        @test length(eb) ==
              length(eb.sampleable_inds) - 1 ==
              length(eb.step_numbers) - 1 ==
              length(eb.episodes_lengths) - 1 ==
              8
        @test last(eb.step_numbers) == 12
        @test size(eb) == size(eb.traces) == (8,)
        empty!(eb)
        @test size(eb) ==
              (0,) ==
              size(eb.traces) ==
              size(eb.sampleable_inds) ==
              size(eb.episodes_lengths) ==
              size(eb.step_numbers)
    end

    @testset "with SARTSA traces and PartialNamedTuple" begin
        eb = EpisodesBuffer(CircularArraySARTSATraces(; capacity = 10))
        # push first episode (five steps)
        push!(eb, (state = 1,))
        @test eb.sampleable_inds[end] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        for i = 1:5
            push!(eb, (state = i + 1, action = i, reward = i, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 0
            if length(eb) >= 1
                @test eb.sampleable_inds[end-2] == 1
            end
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(i, i + 1)
        end
        @test eb.sampleable_inds == [1, 1, 1, 1, 0, 0]
        push!(eb, PartialNamedTuple((action = 6,)))
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0]
        @test length(eb.traces) == 5

        # start second episode
        push!(eb, (state = 7,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 0]
        @test eb[5][:next_action] == eb[:next_action][5] == 6
        @test eb[:reward][6] == 0 # 6 is not a valid index, the reward there is dummy, filled as zero
        @test_throws BoundsError eb[6]  # 6 is not a valid index
        ep2_len = 0
        # push four steps of second episode
        for (i, s) in enumerate(8:11)
            ep2_len += 1
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 0
            if eb.step_numbers[end] > 2
                @test eb.sampleable_inds[end-2] == 1
            end
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
        @test length(eb.traces) == 9 # an action is missing at this stage
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        if eb.step_numbers[end] > 2
            @test eb.sampleable_inds[end-2] == 1
        end

        # push two more steps of second episode, which replace the oldest steps in the buffer
        for (i, s) in enumerate(12:13)
            ep2_len += 1
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 0
            if eb.step_numbers[end] > 2
                @test eb.sampleable_inds[end-2] == 1
            end
            @test eb.step_numbers[end] == i + 1 + 4
            @test eb.episodes_lengths[end-ep2_len:end] == fill(ep2_len, ep2_len + 1)
        end
        push!(eb, PartialNamedTuple((action = 13,)))
        @test length(eb.traces) == 10

        # verify episode 2
        for (i, s) in enumerate(3:13)
            if i in (4, 11)
                @test eb.sampleable_inds[i] == 0
                continue
            else
                @test eb.sampleable_inds[i] == 1
            end
            b = eb[i]
            @test b[:state] == b[:action] == b[:reward] == s
            @test b[:next_state] == b[:next_action] == s + 1
        end

        # push third episode
        push!(eb, (state = 14,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        # push until it reaches it own start
        for (i, s) in enumerate(15:26)
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
        end
        push!(eb, PartialNamedTuple((action = 26,)))
        @test eb.sampleable_inds == [fill(true, 10); [false]]
        @test eb.episodes_lengths == fill(length(15:26), 11)
        @test eb.step_numbers == [3:13;]
        step = popfirst!(eb)
        @test length(eb) ==
              length(eb.sampleable_inds) - 1 ==
              length(eb.step_numbers) - 1 ==
              length(eb.episodes_lengths) - 1 ==
              9
        @test first(eb.step_numbers) == 4
        step = pop!(eb)
        @test length(eb) ==
              length(eb.sampleable_inds) - 1 ==
              length(eb.step_numbers) - 1 ==
              length(eb.episodes_lengths) - 1 ==
              8
        @test last(eb.step_numbers) == 12
        @test size(eb) == size(eb.traces) == (8,)
        empty!(eb)
        @test size(eb) ==
              (0,) ==
              size(eb.traces) ==
              size(eb.sampleable_inds) ==
              size(eb.episodes_lengths) ==
              size(eb.step_numbers)
    end
    @testset "with vector traces" begin
        eb = EpisodesBuffer(Traces(; state = Int[], reward = Int[]))
        push!(eb, (state = 1,)) # partial inserting
        for i = 1:15
            push!(eb, (state = i + 1, reward = i))
        end
        @test length(eb.traces) == 15
        @test eb.sampleable_inds == [fill(true, 15); [false]]
        @test all(==(15), eb.episodes_lengths)
        @test eb.step_numbers == [1:16;]
        push!(eb, (state = 1,)) # partial inserting
        for i = 1:15
            push!(eb, (state = i + 1, reward = i))
        end
        @test eb.sampleable_inds == [fill(true, 15); [false]; fill(true, 15); [false]]
        @test all(==(15), eb.episodes_lengths)
        @test eb.step_numbers == [1:16; 1:16]
        @test length(eb) == 31
    end

    @testset "with ElasticArraySARTSTraces" begin
        eb = EpisodesBuffer(ElasticArraySARTSTraces())
        # push first episode (five steps)
        push!(eb, (state = 1,))
        @test eb.sampleable_inds[end] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        for i = 1:5
            push!(eb, (state = i + 1, action = i, reward = i, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(i, i + 1)
        end
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0]
        @test length(eb.traces) == 5

        # start second episode
        push!(eb, (state = 7,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 0]
        @test eb[:reward][6] == 0 #6 is not a valid index, the reward there is dummy, filled as zero
        @test_throws BoundsError eb[6]  #6 is not a valid index
        ep2_len = 0
        # push four steps of second episode
        for (j, i) in enumerate(8:11)
            ep2_len += 1
            push!(eb, (state = i, action = i - 1, reward = i - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == j + 1
            @test eb.episodes_lengths[end-j:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        @test length(eb.traces) == 10

        # push two more steps of second episode, which replace the oldest steps in the buffer
        for (i, s) in enumerate(12:13)
            ep2_len += 1
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1 + 4
            @test eb.episodes_lengths[end-ep2_len:end] == fill(ep2_len, ep2_len + 1)
        end
        # verify episode 2
        for i = 3:13
            if i in (6, 13)
                @test eb.sampleable_inds[i] == 0
                continue
            else
                @test eb.sampleable_inds[i] == 1
            end
            b = eb[i]
            @test b[:state] == b[:action] == b[:reward] == i
            @test b[:next_state] == i + 1
        end

        # push third episode
        push!(eb, (state = 14,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1

        # push until it reaches it own start
        for (i, s) in enumerate(15:26)
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
        end
        @test eb.sampleable_inds[end-5:end] == [fill(true, 5); [false]]
        @test eb.episodes_lengths[end-10:end] == fill(length(15:26), 11)
        @test eb.step_numbers[end-10:end] == [3:13;]
        #= Deactivated until https://github.com/JuliaArrays/ElasticArrays.jl/pull/56/files merged and pop!/popfirst! added to ElasticArrays
        step = popfirst!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 9
        @test first(eb.step_numbers) == 4
        step = pop!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 8
        @test last(eb.step_numbers) == 12
        @test size(eb) == size(eb.traces) == (8,)
        empty!(eb)
        @test size(eb) == (0,) == size(eb.traces) == size(eb.sampleable_inds) == size(eb.episodes_lengths) == size(eb.step_numbers)
        =#
    end

    @testset "ElasticArraySARTSATraces with PartialNamedTuple" begin
        eb = EpisodesBuffer(ElasticArraySARTSATraces())
        # push first episode (five steps)
        push!(eb, (state = 1,))
        @test eb.sampleable_inds[end] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        for i = 1:5
            push!(eb, (state = i + 1, action = i, reward = i, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 0
            if eb.step_numbers[end] > 2
                @test eb.sampleable_inds[end-2] == 1
            end
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(i, i + 1)
        end
        push!(eb, PartialNamedTuple((action = 6,)))
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0]
        @test length(eb.traces) == 5

        # start second episode
        push!(eb, (state = 7,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 0]
        @test eb[:action][6] == 6
        @test eb[:next_action][5] == 6
        @test eb[:reward][6] == 0 #6 is not a valid index, the reward there is dummy, filled as zero
        ep2_len = 0
        # push four steps of second episode
        for (j, i) in enumerate(8:11)
            ep2_len += 1
            push!(eb, (state = i, action = i - 1, reward = i - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 0
            if eb.step_numbers[end] > 2
                @test eb.sampleable_inds[end-2] == 1
            end
            @test eb.step_numbers[end] == j + 1
            @test eb.episodes_lengths[end-j:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb.sampleable_inds == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
        @test length(eb.traces) == 9 # an action is missing at this stage
        # push two more steps of second episode, which replace the oldest steps in the buffer
        for (i, s) in enumerate(12:13)
            ep2_len += 1
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 0
            if eb.step_numbers[end] > 2
                @test eb.sampleable_inds[end-2] == 1
            end
            @test eb.step_numbers[end] == i + 1 + 4
            @test eb.episodes_lengths[end-ep2_len:end] == fill(ep2_len, ep2_len + 1)
        end
        push!(eb, PartialNamedTuple((action = 13,)))
        @test length(eb.traces) == 12

        # verify episode 2
        for i = 1:13
            if i in (6, 13)
                @test eb.sampleable_inds[i] == 0
                continue
            else
                @test eb.sampleable_inds[i] == 1
            end
            b = eb[i]
            @test b[:state] == b[:action] == b[:reward] == i
            @test b[:next_state] == b[:next_action] == i + 1
        end

        # push third episode
        push!(eb, (state = 14,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        #push until it reaches it own start
        for (i, s) in enumerate(15:26)
            push!(eb, (state = s, action = s - 1, reward = s - 1, terminal = false))
        end
        push!(eb, PartialNamedTuple((action = 26,)))
        @test eb.sampleable_inds[end-10:end] == [fill(true, 10); [false]]
        @test eb.episodes_lengths[end-10:end] == fill(length(15:26), 11)
        @test eb.step_numbers[end-10:end] == [3:13;]
        #= Deactivated until https://github.com/JuliaArrays/ElasticArrays.jl/pull/56/files merged and pop!/popfirst! added to ElasticArrays
        step = popfirst!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 9
        @test first(eb.step_numbers) == 4
        step = pop!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 8
        @test last(eb.step_numbers) == 12
        @test size(eb) == size(eb.traces) == (8,)
        empty!(eb)
        @test size(eb) == (0,) == size(eb.traces) == size(eb.sampleable_inds) == size(eb.episodes_lengths) == size(eb.step_numbers)
        =#
    end
end
