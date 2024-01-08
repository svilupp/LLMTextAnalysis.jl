using LLMTextAnalysis: nunique
@testset "nunique" begin
    @test nunique([1, 2, 3, 4, 5]) == 5
    @test nunique([1, 1, 1, 1, 1]) == 1
    @test nunique([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) == 5
    @test nunique([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]) == 5
end
