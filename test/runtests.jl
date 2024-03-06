using LLMTextAnalysis
using PromptingTools
const PT = PromptingTools
using Test, Random, LinearAlgebra
import Plots, PlotlyJS, Tables
Plots.plotlyjs();
using Aqua

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(LLMTextAnalysis; ambiguities = false)
    @test isempty(Test.detect_ambiguities(LLMTextAnalysis))
end
@testset "LLMTextAnalysis.jl" begin
    include("utils.jl")
    include("types.jl")
    include("preparation.jl")
    include("topic_modelling.jl")
    include("concept_labeling.jl")
    include("classification.jl")
    include("plotting.jl")
end
