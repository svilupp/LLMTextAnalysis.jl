using LLMTextAnalysis: build_topic, build_clusters!, TopicMetadata
using PromptingTools: TestEchoOpenAISchema

@testset "build_topic" begin
    # corresponds to OpenAI API v1
    response = Dict(:choices => [Dict(:message => Dict(:content => "Topic Name"))],
        :usage => Dict(:total_tokens => 3, :prompt_tokens => 2, :completion_tokens => 1))
    PT.register_model!(;
        name = "mock-aigen",
        schema = TestEchoOpenAISchema(; response, status = 200))

    # Basic Functionality: Test topic building for a given topic index
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))
    index = DocIndex(; docs, embeddings, distances,
        keywords_ids = ones(2, 4), keywords_vocab = ["document", "text"])
    assignments = [1, 1, 2, 2] # Assigning first two docs to topic 1, third to topic 2
    topic_metadata = build_topic(index,
        assignments,
        2;
        model = "mock-aigen",
        num_samples = 0,
        num_keywords = 0,
        add_label = false,
        add_summary = false)
    @test isa(topic_metadata, TopicMetadata)
    @test topic_metadata.topic_level == 2
    @test topic_metadata.topic_idx == 2
    @test topic_metadata.docs_idx == [3, 4]
    @test topic_metadata.center_doc_idx in [1, 2]
    @test topic_metadata.samples_doc_idx == Int[]
    @test topic_metadata.keywords_idx == Int[]
    @test topic_metadata.label == ""
    @test topic_metadata.summary == ""

    # Invalid Topic Index: Test with a topic index not in assignments and other asserts
    @test_throws AssertionError build_topic(index, assignments, 3)
    @test_throws AssertionError build_topic(index,
        assignments,
        2;
        add_label = true,
        label_template = nothing)
    @test_throws AssertionError build_topic(index,
        assignments,
        2; num_samples = -1)
    @test_throws AssertionError build_topic(index,
        assignments,
        2; num_keywords = -1)

    ## Enable all options
    index = DocIndex(; docs, embeddings, distances,
        keywords_ids = ones(2, 4), keywords_vocab = ["document", "text"])
    assignments = [1, 1, 2, 2] # Assigning first two docs to topic 1, third to topic 2
    topic_metadata = build_topic(index,
        assignments,
        2;
        model = "mock-aigen",
        add_label = true,
        num_keywords = 1,
        add_summary = true)
    @test isa(topic_metadata, TopicMetadata)
    @test topic_metadata.topic_level == 2
    @test topic_metadata.topic_idx == 2
    @test topic_metadata.docs_idx == [3, 4]
    @test topic_metadata.center_doc_idx in [1, 2]
    @test topic_metadata.samples_doc_idx ==
          [setdiff([1, 2], [topic_metadata.center_doc_idx])...]
    @test topic_metadata.keywords_idx == [1] || topic_metadata.keywords_idx == [2]
    @test topic_metadata.label == "Topic Name"
    @test topic_metadata.summary == "Topic Name"

    # Custom Topic Level: Test with a specified topic level
    custom_topic_level = 5
    topic_metadata_custom_level = build_topic(index,
        assignments,
        1; model = "mock-aigen",
        topic_level = custom_topic_level)
    @test topic_metadata_custom_level.topic_level == custom_topic_level
    ## Clean up
    haskey(PT.MODEL_REGISTRY, "mock-aigen") && delete!(PT.MODEL_REGISTRY, "mock-aigen")
end

@testset "build_clusters!" begin
    # corresponds to OpenAI API v1
    response = Dict(:choices => [Dict(:message => Dict(:content => "Topic Name"))],
        :usage => Dict(:total_tokens => 3, :prompt_tokens => 2, :completion_tokens => 1))
    PT.register_model!(;
        name = "mock-aigen",
        schema = TestEchoOpenAISchema(; response, status = 200))

    # Basic Functionality: Test cluster building with a default k value
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = hcat(ones(Float32, (10, 2)),
        vcat(ones(Float32, (5, 2)), 2ones(Float32, (5, 2))))
    distances = zeros(Float32, (4, 4))
    distances[3:4, 1:2] .= 1
    distances[1:2, 3:4] .= 1
    index = DocIndex(; docs, embeddings, distances,
        keywords_ids = ones(2, 4), keywords_vocab = ["document", "text"])

    @test isempty(index.topic_levels)
    @test index.clustering == nothing
    build_clusters!(index;
        k = 2,
        labeler_kwargs = (; model = "mock-aigen"),
        add_summary = true)
    @test index.clustering isa LLMTextAnalysis.Clustering.Hclust
    @test index.topic_levels[2] isa Vector{TopicMetadata}

    topic = index.topic_levels[2][1]
    @test size(topic.docs_idx, 1) == 2
    @test topic.label == "Topic Name"
    @test topic.summary == "Topic Name"

    ## add one more level
    build_clusters!(index;
        k = 3,
        labeler_kwargs = (; model = "mock-aigen"),
        add_summary = true)
    @test length(index.topic_levels) == 2
    @test index.topic_levels[3] isa Vector{TopicMetadata}

    topic = index.topic_levels[3][1]
    @test size(topic.docs_idx, 1) <= 2
    @test topic.label == "Topic Name"
    @test topic.summary == "Topic Name"

    # Specific h value: Test with a specified height h for dendrogram cutting
    index = DocIndex(; docs, embeddings, distances,
        keywords_ids = ones(2, 4), keywords_vocab = ["document", "text"])

    @test isempty(index.topic_levels)
    @test index.clustering == nothing
    h_value = 0.5
    build_clusters!(index; h = h_value, labeler_kwargs = (; model = "mock-aigen"))
    @test !isempty(index.topic_levels)
    @test length(index.topic_levels) == 1

    # Clustering Already Exists: Ensure existing clustering is used if present
    index = DocIndex(; docs, embeddings, distances,
        keywords_ids = ones(2, 4), keywords_vocab = ["document", "text"],
        topic_levels = Dict(2 => fill(TopicMetadata(;
                index_id = :x,
                topic_idx = 1,
                topic_level = 2),
            2)))
    index.clustering = :precomputed_clustering
    pre_clustered_index = deepcopy(index)
    build_clusters!(index; labeler_kwargs = (; model = "mock-aigen"))
    @test pre_clustered_index.clustering == index.clustering

    ## Clean up
    haskey(PT.MODEL_REGISTRY, "mock-aigen") && delete!(PT.MODEL_REGISTRY, "mock-aigen")
end
