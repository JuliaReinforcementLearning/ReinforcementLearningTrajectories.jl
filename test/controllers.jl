import ReinforcementLearningTrajectories: on_insert!, on_sample!
@testset "controllers.jl" begin
    @testset "EpisodeSampleRatioController" begin
        #push
        c = EpisodeSampleRatioController(ratio = 1 / 2, threshold = 5)
        for st = 1:50
            transition = (state = 1, action = 2, reward = 5.0, terminal = (st % 5 == 0))
            on_insert!(c, 1, transition)
            if st in 25:10:45
                @test on_sample!(c)
                @test !on_sample!(c)
            else
                @test !on_sample!(c)
            end
        end
        #append
        c = EpisodeSampleRatioController(ratio = 1 / 2, threshold = 5)
        for e = 1:20
            transitions = (
                state = ones(5),
                action = ones(5),
                reward = ones(5),
                terminal = [false, false, false, false, iseven(e)],
            )
            on_insert!(c, length(first(transitions)), transitions)
            if e in 10:4:20
                @test on_sample!(c)
                @test !on_sample!(c)
            else
                @test !on_sample!(c)
            end
        end
        c = EpisodeSampleRatioController(ratio = 1 / 4, threshold = 5)
        for e = 1:10
            transitions = (
                state = ones(10),
                action = ones(10),
                reward = ones(10),
                terminal = [
                    false,
                    false,
                    false,
                    false,
                    true,
                    false,
                    false,
                    false,
                    false,
                    true,
                ],
            )
            on_insert!(c, length(first(transitions)), transitions)
            if e in 3:2:10
                @test on_sample!(c)
                @test !on_sample!(c)
            else
                @test !on_sample!(c)
            end
        end
    end
end
