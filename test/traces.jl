@testset "Traces" begin
    t = Traces(;
        a=[1, 2],
        b=Bool[0, 1]
    )

    @test length(t) == 2

    push!(t, (; a=3, b=true))

    @test t[:a][end] == 3
    @test t[:b][end] == true

    append!(t, Traces(a=[4, 5], b=[false, false]))
    @test length(t[:a]) == 5
    @test t[:b][end-1:end] == [false, false]

    @test t[1] == (a=1, b=false)

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

    push!(t, (; state=1))
    push!(t, (; next_state=2))

    @test t[:state] == [1]
    @test t[:next_state] == [2]
    @test t[1] == (state=1, next_state=2)

    append!(t, (; state=[3, 4]))

    @test t[:state] == [1, 2, 3]
    @test t[:next_state] == [2, 3, 4]
    @test t[end] == (state=3, next_state=4)

    pop!(t)
    t[end] == (state=2, next_state=3)
    empty!(t)
    @test length(t) == 0
end

@testset "MergedTraces" begin
    t1 = Traces(a=Int[])
    t2 = Traces(b=Bool[])

    t3 = t1 + t2
    @test t3[:a] === t1[:a]
    @test t3[:b] === t2[:b]

    push!(t3, (; a=1, b=false))
    @test length(t3) == 1
    @test t3[1] == (a=1, b=false)

    append!(t3, Traces(; a=[2, 3], b=[false, true]))
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

    push!(t5, (m=1.0, n=1.0, a=1, b=1))
    @test length(t5) == 1

    push!(t5, (m=2.0, a=2, b=0))

    @test t5[end] == (m=1.0, n=2.0, b=false, a=2)

    t6 = Traces(aa=Int[])
    t7 = Traces(bb=Bool[])
    t8 = (t1 + t2) + (t6 + t7)

    empty!(t8)
    push!(t8, (a=1, b=false, aa=1, bb=false))
    append!(t8, Traces(a=[2, 3], b=[true, true], aa=[2, 3], bb=[true, true]))

    @test length(t8) == 3

    t8_view = t8[2:3]
    t8_view.a[1] = 0
    @test t8[:a][2] == 0
end

@testset "Episode" begin
    t = Episode(
        Traces(
            state=Int[],
            action=Float64[]
        )
    )

    @test length(t) == 0

    push!(t, (state=1, action=1.0))
    @test length(t) == 1

    append!(t, Traces(state=[2, 3], action=[2.0, 3.0]))
    @test length(t) == 3

    @test t[:state] == [1, 2, 3]
    @test t[end-1:end] == (
        state=[2, 3],
        action=[2.0, 3.0]
    )

    t[] = true # seal
    @test_throws ArgumentError push!(t, (state=4, action=4.0))

    pop!(t)
    @test length(t) == 2

    push!(t, (state=4, action=4.0))
    @test length(t) == 3

    t[] = true # seal
    empty!(t)

    @test length(t) == 0
end

@testset "Episodes" begin
    t = Episodes() do
        Episode(Traces(state=Float64[], action=Int[]))
    end

    @test length(t) == 0

    push!(t, (state=1.0, action=1))

    @test length(t) == 1
    @test t[1] == (state=1.0, action=1)

    t[] = true # seal

    push!(t, (state=2.0, action=2))
    @test length(t) == 2

    @test t[end] == (state=2.0, action=2)

    @test t[1:2] == (state=[1.0, 2.0], action=[1, 2])

    push!(t, (state=3.0, action=3))
    t[] = true # seal

    # a vector of episode-level partitions is returned for now
    @test_broken size(t[:state]) == (3,)

    push!(t, Episode(Traces(state=[4.0, 5.0, 6.0], action=[4, 5, 6])))
    @test t[] == false

    t[] = true
    @test length(t) == 6
end
