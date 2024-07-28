using LLMTextAnalysis: nunique, sigmoid, wrap_string

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

@testset "wrap_string" begin
    @test wrap_string("", 10) == ""
    @test wrap_string("Hi", 10) == "Hi"
    @test wrap_string(strip(" Hi "), 10) == "Hi" # SubString type
    output = wrap_string("This function will wrap words into lines", 10)
    @test maximum(length.(split(output, "\n"))) <= 10
    output = wrap_string("This function will wrap words into lines", 20)
    @test_broken maximum(length.(split(output, "\n"))) <= 20 #bug, it adds back the separator
    str = "This function will wrap words into lines"
    @test wrap_string(str, length(str)) == str
    # Unicode testing
    long_unicode_sentence = "Überraschenderweise ℕ𝕖𝕦𝕣𝕠𝕥𝕣𝕒𝕟𝕤𝕞𝕚𝕥𝕥𝕖𝕣 ℂ𝕙𝕣𝕪𝕤𝕒𝕟𝕥𝕙𝕖𝕞𝕦𝕞𝕤 𝕊𝕪𝕟𝕔𝕙𝕣𝕠𝕡𝕙𝕒𝕤𝕠𝕥𝕣𝕠𝕟 Ξ𝕩𝕥𝕣𝕒𝕠𝕣𝕕𝕚𝕟𝕒𝕚𝕣𝕖"
    wrapped = wrap_string(long_unicode_sentence, 20)
    @test all(length ≤ 20 for line in split(wrapped, "\n"))
    @test join(split(wrapped, "\n"), "") == replace(long_unicode_sentence, " " => "")
end