using LLMTextAnalysis: build_keywords
using PromptingTools: TestEchoOpenAISchema

@testset "build_keywords" begin
    # Basic Input: Test with simple input documents
    docs = ["This is a sample document.", "Another sample document.", "x"]
    keywords_ids, keywords_vocab = build_keywords(docs)
    @test keywords_vocab == ["document", "sampl"]
    @test keywords_ids |> Matrix == hcat(0.5ones(2, 2), zeros(2, 1))

    # No Documents: Ensure it throws an error for empty document list
    docs = String[]
    @test_throws AssertionError build_keywords(docs)

    # Custom Stopwords: Test keyword extraction with custom stopwords
    docs = ["A simple example with custom stopwords."]
    custom_stopwords = ["with", "a", "the", "custom"]
    _, keywords_vocab = build_keywords(docs, stopwords = custom_stopwords)
    @test all(!in(Set(custom_stopwords)), keywords_vocab)

    # Minimum Length Constraint: Test keyword extraction with minimum word length
    docs = ["short and long words with supercalifragilisticexpialidocious"]
    min_length = 6
    _, keywords_vocab = build_keywords(docs; min_length)
    @test all(length(word) >= min_length for word in keywords_vocab)

    # Different Return Type: Test with Symbol as return type for keywords
    docs = ["Testing different return type."]
    _, keywords_vocab = build_keywords(docs, SubString{String})
    @test all(x -> isa(x, SubString{String}), keywords_vocab)
end

@testset "build_index" begin
    # Basic Functionality: Test with simple input documents
    # corresponds to OpenAI API v1
    response1 = Dict(:data => [Dict(:embedding => ones(128)), Dict(:embedding => ones(128))],
        :usage => Dict(:total_tokens => 2, :prompt_tokens => 2, :completion_tokens => 0))
    PT.register_model!(;
        name = "mock-emb",
        schema = TestEchoOpenAISchema(; response = response1, status = 200))
    docs = ["Document 1 text", "Document 2 text"]
    index = build_index(docs; aiembed_kwargs = (; model = "mock-emb"))
    @test isa(index, DocIndex)
    @test length(index.docs) == length(docs)
    @test size(index.embeddings, 2) == length(docs)
    @test eltype(index.embeddings) == Float32
    @test index.distances == zeros(2, 2)
    @test index.keywords_ids == 0.5ones(2, 2)
    @test index.keywords_vocab == ["document", "text"]

    # No Documents: Ensure it throws an error for empty document list
    docs = String[]
    @test_throws AssertionError build_index(docs)

    # Verbose Flag: Test with verbose flag enabled
    docs = ["Document 1 text", "Document 2 text"]
    index = build_index(docs, verbose = true, aiembed_kwargs = (; model = "mock-emb"))
    @test_logs (:info, r"Embedding 2 documents") (:info, r"Done embedding.") (:info,
        r"Computing pairwise distances") (:info, r"Extracting keywords") match_mode=:any build_index(docs,
        verbose = true,
        aiembed_kwargs = (; model = "mock-emb"))

    # Custom Index ID: Test with a custom index ID
    custom_id = :CustomIndexID
    index = build_index(docs, index_id = custom_id, aiembed_kwargs = (; model = "mock-emb"))
    @test index.id == custom_id

    ## Clean up
    haskey(PT.MODEL_REGISTRY, "mock-emb") && delete!(PT.MODEL_REGISTRY, "mock-emb")
end

@testset "prepare_plot!" begin
    # Basic Functionality: Test plot preparation on a simple document index
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))
    index = DocIndex(; docs, embeddings, distances)
    @test index.plot_data == nothing
    prepared_index = prepare_plot!(index; n_neighbors = 1)
    @test index === prepared_index
    @test !isnothing(prepared_index.plot_data)
    @test size(prepared_index.plot_data) == (2, length(docs))  # UMAP should reduce to 2 dimensions

    # Verbose Flag: Test with verbose flag enabled
    index = DocIndex(; docs, embeddings, distances)
    @test_logs (:info, r"Computing UMAP") match_mode=:any prepare_plot!(index,
        n_neighbors = 1,
        verbose = true)
    @test !isnothing(index.plot_data)

    # Repeated Calls: Ensure repeated calls do not change plot data
    plot_data = index.plot_data |> copy
    prepare_plot!(index; n_neighbors = 1)
    prepare_plot!(index; n_neighbors = 1)
    @test index.plot_data == plot_data
end
