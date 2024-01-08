@testset "build_topic" begin
    # Basic Functionality: Test topic building for a given topic index
    docs = ["Document 1", "Document 2", "Document 3"]
    index = build_index(docs)
    assignments = [1, 1, 2] # Assigning first two docs to topic 1, third to topic 2
    topic_metadata = build_topic(index, assignments, 1)
    @test isa(topic_metadata, TopicMetadata)
    @test topic_metadata.topic_idx == 1
    @test length(topic_metadata.docs_idx) == 2

    # Invalid Topic Index: Test with a topic index not in assignments
    @test_throws AssertionError build_topic(index, assignments, 3)

    # No Label and Summary: Test topic building without generating label and summary
    topic_metadata_no_label_summary = build_topic(index,
        assignments,
        1,
        add_label = false,
        add_summary = false)
    @test topic_metadata_no_label_summary.label == ""
    @test topic_metadata_no_label_summary.summary == ""

    # Custom Topic Level: Test with a specified topic level
    custom_topic_level = 5
    topic_metadata_custom_level = build_topic(index,
        assignments,
        1,
        topic_level = custom_topic_level)
    @test topic_metadata_custom_level.topic_level == custom_topic_level

    # AI Mock: Ignore AI-related calls with a mock (Assuming a suitable mock function or data is provided)
    # Here we assume `mock_ai_generate` is a mock function that returns a fixed string
    mock_ai_generate = (; args...) -> "Mocked AI Response"
    topic_metadata_ai_mock = build_topic(index,
        assignments,
        1,
        aikwargs = (model = mock_ai_generate,))
    @test topic_metadata_ai_mock.label == "Mocked AI Response"
    @test topic_metadata_ai_mock.summary == "Mocked AI Response"
end

@testset "build_clusters!" begin
    # Basic Functionality: Test cluster building with a default k value
    docs = ["Document 1", "Document 2", "Document 3"]
    index = build_index(docs)
    clustered_index = build_clusters!(index)
    @test isa(clustered_index, AbstractDocumentIndex)
    @test !isempty(clustered_index.topic_levels)

    # Specific k value: Test with a specified k for clustering
    k_value = 2
    clustered_index_k = build_clusters!(index, k = k_value)
    @test length(clustered_index_k.topic_levels[k_value]) == k_value

    # Specific h value: Test with a specified height h for dendrogram cutting
    h_value = 0.5
    clustered_index_h = build_clusters!(index, h = h_value)
    @test !isempty(clustered_index_h.topic_levels)

    # Verbose Flag: Test with verbose flag enabled
    clustered_index_verbose = build_clusters!(index, verbose = true)
    @test isa(clustered_index_verbose, AbstractDocumentIndex)

    # Clustering Already Exists: Ensure existing clustering is used if present
    pre_clustered_index = deepcopy(index)
    pre_clustered_index.clustering = "Precomputed Clustering"
    clustered_index_existing = build_clusters!(pre_clustered_index)
    @test clustered_index_existing.clustering == "Precomputed Clustering"
end
