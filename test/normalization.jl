using Test
using ReinforcementLearningTrajectories
import OnlineStats: mean, std

@testset "normalization.jl" begin
    t = CircularArraySARTSTraces(capacity = 10, state = Float64 => (5,))
    nt = NormalizedTraces(t, reward = scalar_normalizer(), state = array_normalizer((5,)))
    traj = Trajectory(
        container = nt,
        sampler = BatchSampler(1000),
        controller = InsertSampleRatioController(ratio = Inf, threshold = 0),
    )
    m = mean(0:4)
    s = std(0:4)
    ss = std([0, 1, 2, 2, 3, 4])
    push!(traj, (state = fill(m, 5), action = 1)) #this also updates state moments
    for i = 0:4
        r = ((1.0:5.0) .+ i) .% 5
        push!(traj, (state = [r;], action = 1, reward = Float32(i), terminal = false))
    end

    @test mean(nt.normalizers[:reward].os) == m && std(nt.normalizers[:reward].os) == s
    @test all(nt.normalizers[:state].os) do moments
        mean(moments) ≈ m && std(moments) ≈ ss
    end

    unnormalized_batch = t[[1:5;]]
    @test unnormalized_batch[:reward] == [0:4;]
    @test extrema(unnormalized_batch[:state]) == (0, 4)
    normalized_batch = nt[[1:5;]]

    normalized_batch = sample(traj)
    @test all(extrema(normalized_batch[:state]) .≈ ((0, 4) .- m) ./ ss)
    @test all(extrema(normalized_batch[:next_state]) .≈ ((0, 4) .- m) ./ ss)
    @test all(extrema(normalized_batch[:reward]) .≈ ((0, 4) .- m) ./ s)
    #check for no mutation
    unnormalized_batch = t[[1:5;]]
    @test unnormalized_batch[:reward] == [0:4;]
    @test extrema(unnormalized_batch[:state]) == (0, 4)
end
