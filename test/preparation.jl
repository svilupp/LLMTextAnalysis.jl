@testset "build_keywords" begin
    # Basic Input: Test with simple input documents
    docs = ["This is a sample document.", "Another sample document."]
    keywords_ids, keywords_vocab = build_keywords(docs)
    @test isa(keywords_ids, SparseMatrixCSC)
    @test isa(keywords_vocab, Vector)

    # No Documents: Ensure it throws an error for empty document list
    docs = []
    @test_throws AssertionError build_keywords(docs)

    # Custom Stopwords: Test keyword extraction with custom stopwords
    docs = ["A simple example with custom stopwords."]
    custom_stopwords = ["with", "a", "the"]
    _, keywords_vocab = build_keywords(docs, stopwords = custom_stopwords)
    @test all(!in(Set(custom_stopwords)), keywords_vocab)

    # Minimum Length Constraint: Test keyword extraction with minimum word length
    docs = ["Short and long words"]
    min_length = 5
    _, keywords_vocab = build_keywords(docs, min_length = min_length)
    @test all(length(word) >= min_length for word in keywords_vocab)

    # Different Return Type: Test with Symbol as return type for keywords
    docs = ["Testing different return type."]
    keywords_ids, _ = build_keywords(docs, return_type = Symbol)
    @test all(x -> isa(x, Symbol), nonzeros(keywords_ids))
end

@testset "build_index" begin
    # Basic Functionality: Test with simple input documents
    docs = ["Document 1 text", "Document 2 text"]
    index = build_index(docs)
    @test isa(index, DocIndex)
    @test length(index.docs) == length(docs)
    @test size(index.embeddings, 2) == length(docs)

    # No Documents: Ensure it throws an error for empty document list
    docs = []
    @test_throws AssertionError build_index(docs)

    # Verbose Flag: Test with verbose flag enabled
    docs = ["Document 1 text", "Document 2 text"]
    index = build_index(docs, verbose = true)
    @test isa(index, DocIndex)

    # Custom Index ID: Test with a custom index ID
    custom_id = :CustomIndexID
    index = build_index(docs, index_id = custom_id)
    @test index.id == custom_id

    # AI Embedding Mock: Ignore AI embedding calls with a mock
    # (Assuming a suitable mock function or data is provided)
    mock_embeddings = rand(Float32, (10, 2))
    mock_aiembed = (; docs...) -> mock_embeddings
    index = build_index(docs, aiembed_kwargs = (model = mock_aiembed,))
    @test size(index.embeddings) == size(mock_embeddings)
end

@testset "prepare_plot!" begin
    # Basic Functionality: Test plot preparation on a simple document index
    docs = ["Document 1 text", "Document 2 text"]
    index = build_index(docs)
    prepared_index = prepare_plot!(index)
    @test isa(prepared_index, AbstractDocumentIndex)
    @test !isnothing(prepared_index.plot_data)
    @test size(prepared_index.plot_data, 1) == 2  # UMAP should reduce to 2 dimensions

    # Verbose Flag: Test with verbose flag enabled
    prepared_index_verbose = prepare_plot!(index, verbose = true)
    @test isa(prepared_index_verbose, AbstractDocumentIndex)
    @test !isnothing(prepared_index_verbose.plot_data)

    # Repeated Calls: Ensure repeated calls do not change plot data
    repeated_prepared_index = prepare_plot!(index)
    @test prepared_index.plot_data == repeated_prepared_index.plot_data

    # Plot Data Already Exists: Test when plot data is pre-generated
    mock_plot_data = rand(Float32, (2, length(docs)))
    index.plot_data = mock_plot_data
    prepared_index_with_data = prepare_plot!(index)
    @test prepared_index_with_data.plot_data == mock_plot_data
end
