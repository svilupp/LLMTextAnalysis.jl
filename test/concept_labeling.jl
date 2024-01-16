
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

    # Test for different values of k and lambda
    @test cross_validate_accuracy(Xt, y; k = 3, lambda = 1e-4) isa Float64
    @test cross_validate_accuracy(Xt, y; k = 10, lambda = 0.1) isa Float64

    # Test for error with invalid inputs
    @test_throws AssertionError cross_validate_accuracy(Xt, y; k = -1)
    @test_throws AssertionError cross_validate_accuracy(Xt, y; k = 0, lambda = -0.1)
end

@testset "train_concept" begin
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
    response2 = Dict(:data => [Dict(:embedding => ones(10)) for i in 1:4],
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
    @test concept isa TrainedConcept
    @test concept.concept == "none"
    @test concept.source_doc_ids == [1, 2, 3, 4]
    emb = mapreduce(normalize, hcat, eachcol(ones(Float32, (10, 4)))) .-
          ones(Float32, (10, 4))
    @test concept.embeddings ≈ emb
    @test concept.docs == fill("Topic Name", 4)
    @test !isnothing(concept.coeffs) && length(concept.coeffs) == 10

    # Test 2: Verify if function updates coefficients
    original_pointer = pointer(concept.coeffs)
    trained_concept = train!(index, concept; lambda = 10.0)
    @test pointer(concept.coeffs) != original_pointer

    # Test 3: Check behavior with overwrite = true
    original_emb_pointer = pointer(concept.embeddings)
    train!(index, concept, overwrite = false)
    @test pointer(concept.embeddings) == original_emb_pointer

    train!(index, concept, overwrite = true, aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test pointer(concept.embeddings) != original_emb_pointer

    @test_logs (:info, "Training a classifier...") (:info,
        r"Cross-validated accuracy:") match_mode=:any train!(index,
        concept,
        verbose = true,
        overwrite = false)

    # Test 4: Validate warning for Index ID mismatch
    alt_index = DocIndex(; docs, embeddings, distances)
    @test_throws AssertionError train!(alt_index, concept; overwrite = false)

    # Test 5: Check handling of empty concept
    @test_throws AssertionError train!(index,
        TrainedConcept(; index_id = index.id, source_doc_ids = [1], concept = ""))

    # End-to-end train_concept
    concept = train_concept(index,
        "none"; num_samples = 4,
        aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test concept isa TrainedConcept
    @test concept.concept == "none"
    @test sort(concept.source_doc_ids) == sort([1, 2, 3, 4])
    emb = mapreduce(normalize, hcat, eachcol(ones(Float32, (10, 4)))) .-
          ones(Float32, (10, 4))
    @test concept.embeddings ≈ emb
    @test concept.docs == fill("Topic Name", 4)
    @test !isnothing(concept.coeffs) && length(concept.coeffs) == 10

    ## Clean up
    haskey(PT.MODEL_REGISTRY, "mock-emb") && delete!(PT.MODEL_REGISTRY, "mock-emb")
    haskey(PT.MODEL_REGISTRY, "mock-aigen") && delete!(PT.MODEL_REGISTRY, "mock-aigen")
end

@testset "train_spectrum" begin
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))
    index = DocIndex(; docs, embeddings, distances)

    # Mock AI generation and embedding responses (assuming similar structure as in the original test)
    response = Dict(:choices => [Dict(:message => Dict(:content => "Rewritten Text"))],
        :usage => Dict(:total_tokens => 3, :prompt_tokens => 2, :completion_tokens => 1))
    PT.register_model!(;
        name = "mock-aigen",
        schema = TestEchoOpenAISchema(; response, status = 200))
    response2 = Dict(:data => [Dict(:embedding => ones(10)) for i in 1:4],
        :usage => Dict(:total_tokens => 2, :prompt_tokens => 2, :completion_tokens => 0))
    PT.register_model!(;
        name = "mock-emb",
        schema = TestEchoOpenAISchema(; response = response2, status = 200))

    # Test 1: Check if function returns a TrainedSpectrum object
    spectrum = TrainedSpectrum(;
        index_id = index.id,
        spectrum = ("positive", "negative"),
        source_doc_ids = [1, 2, 3, 4])
    spectrum = train!(index,
        spectrum;
        aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test spectrum isa TrainedSpectrum
    @test spectrum.spectrum == ("positive", "negative")
    @test spectrum.source_doc_ids == [1, 2, 3, 4]

    # Test 2: Verify if function updates coefficients
    original_pointer = pointer(spectrum.coeffs)
    trained_spectrum = train!(index, spectrum; lambda = 0.0)
    @test pointer(spectrum.coeffs) != original_pointer

    # Test 3: Check behavior with overwrite = true
    original_emb_pointer = pointer(spectrum.embeddings)
    train!(index, spectrum, overwrite = false)
    @test pointer(spectrum.embeddings) == original_emb_pointer

    train!(index, spectrum, overwrite = true, aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test pointer(spectrum.embeddings) != original_emb_pointer

    @test_logs (:info, "Training a classifier...") (:info,
        r"Cross-validated accuracy") match_mode=:any train!(index,
        spectrum,
        verbose = true,
        overwrite = false)

    # Test 4: Validate warning for Index ID mismatch
    alt_index = DocIndex(; docs, embeddings, distances)
    @test_throws AssertionError train!(alt_index, spectrum; overwrite = false)

    # Test 5: Check handling of empty spectrum
    @test_throws AssertionError train!(index,
        TrainedSpectrum(;
            index_id = index.id,
            source_doc_ids = [1],
            spectrum = ("", "negative")))

    # End-to-end train_spectrum
    spectrum = train_spectrum(index,
        ("positive", "negative"); num_samples = 4,
        aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test spectrum isa TrainedSpectrum
    @test spectrum.spectrum == ("positive", "negative")
    @test sort(spectrum.source_doc_ids) == sort([1, 2, 3, 4])
    @test !isnothing(spectrum.coeffs) && length(spectrum.coeffs) == 10

    ## Clean up
    haskey(PT.MODEL_REGISTRY, "mock-emb") && delete!(PT.MODEL_REGISTRY, "mock-emb")
    haskey(PT.MODEL_REGISTRY, "mock-aigen") && delete!(PT.MODEL_REGISTRY, "mock-aigen")
end
