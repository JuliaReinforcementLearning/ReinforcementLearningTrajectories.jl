@testset "Traces" begin
    t = Traces(; a = [1, 2], b = Bool[0, 1])

    @test length(t) == 2

    push!(t, (; a = 3, b = true))

    @test t[:a][end] == 3
    @test t[:b][end] == true

    append!(t, Traces(a = [4, 5], b = [false, false]))
    @test length(t[:a]) == 5
    @test t[:b][end-1:end] == [false, false]

    @test t[1] == (a = 1, b = false)

    t_12 = t[1:2]
    @test t_12.a == [1, 2]
    @test t_12.b == [false, true]

    t_12_view = t[1:2]
    t_12_view.a[1] = 0
    @test t[:a][1] == 0

    pop!(t)
    @test length(t) == 4

    popfirst!(t)
    @test length(t) == 3

    empty!(t)
    @test length(t) == 0
end

@testset "MultiplexTraces" begin
    t = MultiplexTraces{(:state, :next_state)}(Int[])

    @test length(t) == 0

    push!(t, (; state = 1))
    push!(t, (; next_state = 2))

    @test t[:state] == [1]
    @test t[:next_state] == [2]
    @test t[1] == (state = 1, next_state = 2)

    append!(t, (; state = [3, 4]))

    @test t[:state] == [1, 2, 3]
    @test t[:next_state] == [2, 3, 4]
    @test t[end] == (state = 3, next_state = 4)

    pop!(t)
    t[end] == (state = 2, next_state = 3)
    empty!(t)
    @test length(t) == 0

    t2 = MultiplexTraces{(:state, :next_state)}(Int[1, 2, 3, 4])
    append!(t, t2[:state])
    @test t[:state] == [1, 2, 3]
    @test t[:next_state] == [2, 3, 4]
end

@testset "MergedTraces" begin
    t1 = Traces(a = Int[])
    t2 = Traces(b = Bool[])

    t3 = t1 + t2
    @test t3[:a] === t1[:a]
    @test t3[:b] === t2[:b]

    push!(t3, (; a = 1, b = false))
    @test length(t3) == 1
    @test t3[1] == (a = 1, b = false)

    append!(t3, Traces(; a = [2, 3], b = [false, true]))
    @test length(t3) == 3

    @test t3[:a][1:3] == [1, 2, 3]

    t3_view = t3[1:3]
    t3_view[:a][1] = 0
    @test t3[:a][1] == 0

    pop!(t3)
    @test length(t3) == 2

    empty!(t3)
    @test length(t3) == 0

    t4 = MultiplexTraces{(:m, :n)}(Float64[])
    t5 = t4 + t2 + t1

    push!(t5, (m = 1.0, n = 1.0, a = 1, b = 1))
    @test length(t5) == 1

    push!(t5, (m = 2.0, a = 2, b = 0))

    @test t5[end] == (m = 1.0, n = 2.0, b = false, a = 2)

    t6 = Traces(aa = Int[])
    t7 = Traces(bb = Bool[])
    t8 = (t1 + t2) + (t6 + t7)

    empty!(t8)
    push!(t8, (a = 1, b = false, aa = 1, bb = false))
    append!(t8, Traces(a = [2, 3], b = [true, true], aa = [2, 3], bb = [true, true]))

    @test length(t8) == 3

    t8_view = t8[2:3]
    t8_view.a[1] = 0
    @test t8[:a][2] == 0
end

using ReinforcementLearningTrajectories: build_trace_index

@testset "build_trace_index" begin
    t1 = CircularArraySARTSATraces(;
        capacity = 3,
        state = Float32 => (2, 3),
        action = Float32 => (2,),
        reward = Float32 => (),
        terminal = Bool => (),
    )
    @test build_trace_index(typeof(t1).parameters[1], typeof(t1).parameters[2]) == Dict(
        :reward => 3,
        :next_state => 1,
        :state => 1,
        :action => 2,
        :next_action => 2,
        :terminal => 4,
    )

    t2 = Traces(; a = [2, 3], b = [false, true])
    build_trace_index(typeof(t2).parameters[1], typeof(t2).parameters[2])
end

@testset "build_trace_index ElasticArraySARTSATraces" begin
    t1 = ElasticArraySARTSATraces(;
        state = Float32 => (2, 3),
        action = Float32 => (2,),
        reward = Float32 => (),
        terminal = Bool => (),
    )
    @test build_trace_index(typeof(t1).parameters[1], typeof(t1).parameters[2]) == Dict(
        :reward => 3,
        :next_state => 1,
        :state => 1,
        :action => 2,
        :next_action => 2,
        :terminal => 4,
    )

    t2 = Traces(; a = [2, 3], b = [false, true])
    build_trace_index(typeof(t2).parameters[1], typeof(t2).parameters[2])
end

@testset "push!(ts::Traces{names,Trs,N,E}, ::Val{k}, v)" begin
    t1 = CircularArraySARTSATraces(;
        capacity = 3,
        state = Float32 => (2, 3),
        action = Float32 => (2,),
        reward = Float32 => (),
        terminal = Bool => (),
    )
    push!(t1, Val(:reward), 5)
    @test t1[:reward][1] == 5

    @test size(Base.getindex(t1, :reward)) == (1,)

    push!(t1, Val(:state), ones(2, 3))

    @test t1[:state][1] == ones(2, 3)

    t2 = Traces(; a = [2, 3], b = [false, true])
    push!(t2, Val(:a), 5)
    @test t2[:a][3] == 5

    @test size(Base.getindex(t2, :a)) == (3,)
    @test Base.getindex(t2, 1) == (; a = 2, b = false)
end


@testset "push!(ts::Traces{names,Trs,N,E}, ::Val{k}, v)" begin
    t1 = ElasticArraySARTSATraces(
        state = Float32 => (2, 3),
        action = Float32 => (2,),
        reward = Float32 => (),
        terminal = Bool => (),
    )
    push!(t1, Val(:reward), 5)
    @test t1[:reward][1] == 5

    @test size(Base.getindex(t1, :reward)) == (1,)
    @test size(Base.getindex(t1, :state)) == (0,)


    t2 = Traces(; a = [2, 3], b = [false, true])
    push!(t2, Val(:a), 5)
    @test t2[:a][3] == 5

    @test size(Base.getindex(t2, :a)) == (3,)
    @test Base.getindex(t2, 1) == (; a = 2, b = false)
end
