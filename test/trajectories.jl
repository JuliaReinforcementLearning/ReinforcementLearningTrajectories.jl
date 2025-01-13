@testset "Default InsertSampleRatioController" begin
    t = Trajectory(container = Traces(a = Int[], b = Bool[]), sampler = BatchSampler(3))
    batches = collect(t)
    @test length(batches) == 0

    push!(t, (a = 1,))
    for i = 1:10
        push!(t, (a = i, b = true))
    end

    batches = collect(t)
    @test length(batches) == 11
end

@testset "trajectories" begin
    t = Trajectory(
        container = Traces(a = Int[], b = Bool[]),
        sampler = BatchSampler(3),
        controller = InsertSampleRatioController(ratio = 0.25, threshold = 4),
    )

    batches = []

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 0  # threshold not reached yet

    push!(t, (a = 1,))
    for i = 1:2
        push!(t, (a = i + 1, b = true))
    end

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 0  # threshold not reached yet

    push!(t, (a = 4, b = true))

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 1  # 4 inserted, threshold is 4, ratio is 0.25

    for i = 5:7
        push!(t, (a = i, b = true))
    end

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 1  # 7 inserted, threshold is 4, ratio is 0.25

    push!(t, (a = 8, b = true))

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 2  # 8 inserted, ratio is 0.25

    n = 400
    for i = 1:n
        push!(t, (a = i, b = true))
    end

    s = 0
    for _ in t
        s += 1
    end
    @test s == n * 0.25
end

@testset "async trajectories" begin
    threshould = 100
    ratio = 1 / 4
    t = Trajectory(
        container = Traces(a = Int[], b = Bool[]),
        sampler = BatchSampler(3),
        controller = AsyncInsertSampleRatioController(ratio, threshould),
    )

    n = 100
    insert_task = @async for i = 1:n
        append!(t, Traces(a = [i, i, i, i], b = [false, true, false, true]))
    end

    s = 0
    sample_task = @async for _ in t
        s += 1
    end
    sleep(1)
    @test s == (n - threshould * ratio) + 1
end
