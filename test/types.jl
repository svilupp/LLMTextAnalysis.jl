@testset "show for TopicMetadata" begin
    # Basic Functionality: Check the output of the show method
    topic_metadata = TopicMetadata(index_id = :test_id,
        topic_level = 2,
        topic_idx = 2,
        label = "Test Label",
        summary = "Test Summary")
    io = IOBuffer()
    show(io, topic_metadata)
    output = String(take!(io))
    @test output ==
          "TopicMetadata(ID: 2/2, Documents: 0, Label: \"Test Label\",  Summary: Available)"

    # No summary, no label
    topic_metadata.label = ""
    topic_metadata.summary = ""
    show(io, topic_metadata)
    output = String(take!(io))
    @test output ==
          "TopicMetadata(ID: 2/2, Documents: 0, Label: -,  Summary: -)"
end

@testset "show for DocIndex" begin
    # Basic Functionality: Check the output of the show method
    docs = ["Doc 1", "Doc 2"]
    embeddings = rand(Float32, (10, 2))
    distances = rand(Float32, (2, 2))
    index = DocIndex(docs = docs, embeddings = embeddings, distances = distances)
    io = IOBuffer()
    show(io, index)
    output = String(take!(io))
    @test output == "DocIndex(Documents: 2, PlotData: None, Topic Levels: None)"

    # With Plot Data: Check output when plot data is present
    index.plot_data = rand(Float32, (2, 2))
    io = IOBuffer()
    show(io, index)
    output = String(take!(io))
    @test output == "DocIndex(Documents: 2, PlotData: OK, Topic Levels: None)"
end
