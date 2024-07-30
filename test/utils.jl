using LLMTextAnalysis: nunique, sigmoid

@testset "nunique" begin
    @test nunique([1, 2, 3, 4, 5]) == 5
    @test nunique([1, 1, 1, 1, 1]) == 1
    @test nunique([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) == 5
    @test nunique([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]) == 5
end

@testset "sigmoid" begin
    # Test for x = 0
    @test sigmoid(0.0)≈0.5 atol=1e-6

    # Test for x approaching positive infinity
    @test sigmoid(Inf)≈1.0 atol=1e-6

    # Test for x approaching negative infinity
    @test sigmoid(-Inf)≈0.0 atol=1e-6

    # Test for other values
    @test sigmoid(1.0) > 0.5 && sigmoid(1.0) < 1.0
    @test sigmoid(-1.0) > 0.0 && sigmoid(-1.0) < 0.5

    # Test to ensure output is between 0 and 1 for any real number
    for x in -100.0:1.0:100.0
        @test 0.0 ≤ sigmoid(x) ≤ 1.0
    end
end