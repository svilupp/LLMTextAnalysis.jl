
using LLMTextAnalysis: create_folds, cross_validate_accuracy
using LLMTextAnalysis: sigmoid, train!
using PromptingTools: TestEchoOpenAISchema

@testset "create_folds" begin
    kfold = 4
    n_obs = 100 # Example data size

    # Test for correct number of folds
    @test length(create_folds(kfold, n_obs)) == kfold

    # Test for correct total number of observations
    @test sum(length.(create_folds(kfold, n_obs))) == n_obs

    # Test for non-empty folds
    @test all(fold -> !isempty(fold), create_folds(kfold, n_obs))

    # Test for handling non-divisible cases
    @test length(create_folds(kfold, n_obs + 1)) == 4

    # Test for error with invalid inputs
    @test_throws AssertionError create_folds(0, n_obs)
    @test_throws AssertionError create_folds(4, 2)
end

@testset "cross_validate_accuracy" begin
    # Assuming some dummy data is available
    Xt = randn(100, 10)
    y = rand([1, -1], 100)

    # Test for return type
    @test cross_validate_accuracy(Xt, y) isa Float64

    # Test for accuracy range
    @test 0.0 <= cross_validate_accuracy(Xt, y) <= 1.0

    # Test for different values of k and λ
    @test cross_validate_accuracy(Xt, y; k = 3, λ = 1e-4) isa Float64
    @test cross_validate_accuracy(Xt, y; k = 10, λ = 0.1) isa Float64

    # Test for error with invalid inputs
    @test_throws AssertionError cross_validate_accuracy(Xt, y; k = -1)
    @test_throws AssertionError cross_validate_accuracy(Xt, y; k = 0, λ = -0.1)
end

# @testset "train!-concept" begin
docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
embeddings = ones(Float32, (10, 4))
distances = zeros(Float32, (4, 4))
index = DocIndex(; docs, embeddings, distances)

# corresponds to OpenAI API v1
response = Dict(:choices => [Dict(:message => Dict(:content => "Topic Name"))],
    :usage => Dict(:total_tokens => 3, :prompt_tokens => 2, :completion_tokens => 1))
PT.register_model!(;
    name = "mock-aigen",
    schema = TestEchoOpenAISchema(; response, status = 200))
response2 = Dict(:data => [Dict(:embedding => ones(10)), Dict(:embedding => ones(10))],
    :usage => Dict(:total_tokens => 2, :prompt_tokens => 2, :completion_tokens => 0))
PT.register_model!(;
    name = "mock-emb",
    schema = TestEchoOpenAISchema(; response = response2, status = 200))

# Test 1: Check if function returns a TrainedConcept object
concept = TrainedConcept(;
    index_id = index.id,
    concept = "none",
    source_doc_ids = [1, 2, 3, 4])
concept = train!(index,
    concept;
    aiembed_kwargs = (; model = "mock-emb"),
    aigenerate_kwargs = (; model = "mock-aigen"))

# Test 2: Verify if function updates coefficients
@test begin
    original_coeffs = copy(mock_concept.coeffs)
    trained_concept = train!(mock_index, mock_concept)
    "Coefficients are updated" != original_coeffs == trained_concept.coeffs
end

# Test 3: Check behavior with overwrite = true
@test begin
    trained_concept = train!(mock_index, mock_concept, overwrite = true)
    "Overwrite flag updates model correctly" == !isempty(trained_concept.coeffs)
end

# Test 4: Validate warning for Index ID mismatch
@test begin
    mock_index_different = ... # Create a different mock AbstractDocumentIndex
    @test_warn "Index ID mismatch" train!(mock_index_different, mock_concept)
end

# Test 5: Check handling of empty concept
@test_throws AssertionError train!(mock_index, TrainedConcept(...; concept = ""))

# Additional tests can be added here as needed
# end
