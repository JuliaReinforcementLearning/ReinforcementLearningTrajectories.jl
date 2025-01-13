function gen_rand_sumtree(n, seed, type::DataType = Float32)
    rng = StableRNG(seed)
    a = SumTree(type, n)
    append!(a, rand(rng, type, n))
    return a
end

function gen_sumtree_with_zeros(n, seed, type::DataType = Float32)
    a = gen_rand_sumtree(n, seed, type)
    b = rand(StableRNG(seed), Bool, n)
    return copy_multiply(a, b)
end

function copy_multiply(stree, m)
    new_tree = deepcopy(stree)
    new_tree .*= m
    return new_tree
end

function sumtree_nozero(t::SumTree, rng::AbstractRNG, iters = 1)
    for _ in iters
        (_, p) = rand(rng, t)
        p == 0 && return false
    end
    return true
end
sumtree_nozero(n::Integer, seed::Integer, iters = 1) =
    sumtree_nozero(gen_sumtree_with_zeros(n, seed), StableRNG(seed), iters)
sumtree_nozero(n, seeds::AbstractVector, iters = 1) =
    all(sumtree_nozero(n, seed, iters) for seed in seeds)


function sumtree_distribution!(
    indices,
    priorities,
    t::SumTree,
    rng::AbstractRNG,
    iters = 1000 * t.length,
)
    for i = 1:iters
        indices[i], priorities[i] = rand(rng, t)
    end
    imap = countmap(indices)
    est_pdf = Dict(k => v / length(indices) for (k, v) in imap)
    ex_pdf = Dict(k => v / t.tree[1] for (k, v) in Dict(1:length(t) .=> t))
    abserrs = [est_pdf[k] - ex_pdf[k] for k in keys(est_pdf)]
    return abserrs
end
sumtree_distribution!(indices, priorities, n, seed, iters = 1000 * n) =
    sumtree_distribution!(
        indices,
        priorities,
        gen_rand_sumtree(n, seed),
        StableRNG(seed),
        iters,
    )
function sumtree_distribution(n, seeds::AbstractVector, iters = 1000 * n)
    p = [zeros(Float32, iters) for _ = 1:Threads.nthreads()]
    i = [zeros(Float32, iters) for _ = 1:Threads.nthreads()]
    results = Vector{Vector{Float64}}(undef, length(seeds))
    Threads.@threads for ix = 1:length(seeds)
        results[ix] = sumtree_distribution!(
            i[Threads.threadid()],
            p[Threads.threadid()],
            gen_rand_sumtree(n, seeds[ix]),
            StableRNG(seeds[ix]),
            iters,
        )
    end
    return results
end

@testset "SumTree" begin
    n = 1024
    seeds = 1:100
    nozero_iters = 1024
    distr_iters = 1024 * 10_000
    abstol = 0.05
    maxerr = 0.01

    @test sumtree_nozero(n, seeds, nozero_iters)
    @test all(
        x -> all(x .< maxerr) && sum(abs2, x) < abstol,
        sumtree_distribution(n, seeds, distr_iters),
    )
end
