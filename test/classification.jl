
using LLMTextAnalysis: score, train_classifier, TrainedClassifier
using LLMTextAnalysis: softmax, train!, CosineDist, pairwise
using PromptingTools: TestEchoOpenAISchema

@testset "train_classifier" begin
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

    ## Data Mock
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    embeddings[1:5, 1:2] .= 2
    distances = pairwise(CosineDist(), embeddings; dims = 2)
    index = DocIndex(; docs, embeddings, distances)
    labels = ["Topic A", "Topic B"]
    docs_ids = [1, 2, 3, 4]
    docs_labels = [1, 1, 2, 2]

    # Test 1: Check if function returns a TrainedClassifier object
    cls = TrainedClassifier(;
        index_id = index.id,
        labels,
        docs = index.docs[docs_ids],
        source_doc_ids = docs_ids,
        embeddings, docs_labels
    )
    cls = train!(index,
        cls; overwrite = false)
    @test cls isa TrainedClassifier
    @test cls.source_doc_ids == [1, 2, 3, 4]
    @test size(cls.coeffs) == (10, 2)

    # Test 2: Verify if function updates coefficients
    original_pointer = pointer(cls.coeffs)
    trained_cls = train!(index, cls; lambda = 10.0)
    @test pointer(trained_cls.coeffs) != original_pointer

    # Test 3: Check behavior with overwrite = true
    original_emb_pointer = pointer(cls.embeddings)
    train!(index, cls, overwrite = false)
    @test pointer(cls.embeddings) == original_emb_pointer

    train!(index, cls, overwrite = true, aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test pointer(cls.embeddings) != original_emb_pointer

    @test_logs (:info, "Training a classifier...") match_mode=:any train!(index,
        cls,
        verbose = true,
        overwrite = false)

    # Test 4: Validate warning for Index ID mismatch
    alt_index = DocIndex(; docs, embeddings, distances)
    @test_throws AssertionError train!(alt_index, cls; overwrite = false)

    # Test 5: Check handling of empty labels
    @test_throws AssertionError train!(index,
        TrainedClassifier(;
            index_id = index.id, docs_labels = [1], source_doc_ids = [1], labels = String[]))

    # End-to-end train_concept
    cls = train_classifier(index,
        labels;
        docs_ids, docs_labels,
        aiembed_kwargs = (; model = "mock-emb"),
        aigenerate_kwargs = (; model = "mock-aigen"))
    @test cls isa TrainedClassifier
    @test cls.labels == labels
    @test cls.embeddings ≈ embeddings
    @test !isnothing(cls.coeffs) && size(cls.coeffs) == (10, 2)

    # Scoring
    scores = score(index, cls)
    @test scores≈[1.0 0; 1.0 0; 0 1.0; 0 1.0] atol=0.1
    # Labels
    best_labels = score(index, cls; return_labels = true)
    @test best_labels == ["Topic A", "Topic A", "Topic B", "Topic B"]
    # wrong index
    alt_index = DocIndex(; docs, embeddings, distances)
    @test_logs (:warn, r"Potential error:") match_mode=:any scores=score(alt_index, cls)

    # functor dispatch
    scores = cls(index)
    @test scores≈[1.0 0; 1.0 0; 0 1.0; 0 1.0] atol=0.1
    # Labels
    best_labels = score(index, cls; return_labels = true)
    @test best_labels == ["Topic A", "Topic A", "Topic B", "Topic B"]

    ## Input validations
    # Too few labels 
    @test_throws AssertionError train_classifier(index,
        ["a"];
        docs_ids, docs_labels)
    @test_throws AssertionError train_classifier(index,
        ["a", "b", "c"];
        docs_ids, docs_labels = ones(Int, 3))
    # mismatch docs vs doc labels
    @test_throws AssertionError train_classifier(index,
        labels;
        docs_ids, docs_labels = ones(Int, 3))
    @test_throws AssertionError train_classifier(index,
        labels;
        docs_ids = ones(Int, 4) * 10, docs_labels)
    # Labels descriptions
    @test_throws AssertionError train_classifier(index,
        labels;
        docs_ids, docs_labels,
        labels_description = fill("", 5))

    ## Clean up
    haskey(PT.MODEL_REGISTRY, "mock-emb") && delete!(PT.MODEL_REGISTRY, "mock-emb")
    haskey(PT.MODEL_REGISTRY, "mock-aigen") && delete!(PT.MODEL_REGISTRY, "mock-aigen")
end
