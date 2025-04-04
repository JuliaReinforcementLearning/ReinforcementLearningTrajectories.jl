# ReinforcementLearningTrajectories

[![Build Status](https://github.com/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/Trajectories.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Design

The relationship of several concepts provided in this package:

```
┌───────────────────────────────────┐
│ Trajectory                        │
│ ┌───────────────────────────────┐ │
│ │ EpisodesBuffer wrapping a     | |
| | AbstractTraces                │ │
│ │             ┌───────────────┐ │ │
│ │ :trace_A => │ AbstractTrace │ │ │
│ │             └───────────────┘ │ │
│ │                               │ │
│ │             ┌───────────────┐ │ │
│ │ :trace_B => │ AbstractTrace │ │ │
│ │             └───────────────┘ │ │
│ │  ...             ...          │ │
│ └───────────────────────────────┘ │
│          ┌───────────┐            │
│          │  Sampler  │            │
│          └───────────┘            │
│         ┌────────────┐            │
│         │ Controller │            │
│         └────────────┘            │
└───────────────────────────────────┘
```

## `Trajectory`

A `Trajectory` contains 3 parts:

- A `container` to store data. (Usually an `AbstractTraces`)
- A `sampler` to determine how to sample a batch from `container`
- A `controller` to decide when to sample a new batch from the `container`

Typical usage:

```julia
julia> t = Trajectory(
               container = Traces(a=Int[], b=Bool[]), 
               sampler = BatchSampler(3), 
               controller = InsertSampleRatioController(1.0, 3, 0, 0)
           );

julia> push!(t, (a=1,));

julia> for i in 1:5
           push!(t, (a=i, b=iseven(i)))
       end

julia> for batch in t
           println(batch)
       end
(a = [1, 3, 1], b = Bool[1, 1, 1])
(a = [4, 1, 4], b = Bool[0, 0, 0])
(a = [1, 4, 1], b = Bool[1, 0, 0])
(a = [1, 1, 4], b = Bool[1, 0, 0])
```

**Traces**

- `Traces`
- `MultiplexTraces`
- `CircularSARTTraces`
- `NormalizedTraces`

**Samplers**

- `BatchSampler`
- `MetaSampler`
- `MultiBatchSampler`
- `EpisodesSampler`

**Controllers**

- `InsertSampleRatioController` 
- `AsyncInsertSampleRatioController`


Please refer tests for common usage. (TODO: generate docs and add links to above data structures)

## Acknowledgement

This async version is mainly inspired by [deepmind/reverb](https://github.com/deepmind/reverb). 
