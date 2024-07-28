@testset "show-TopicMetadata" begin
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

@testset "show-DocIndex" begin
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

@testset "TrainedConcept" begin
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))
    index = DocIndex(; docs, embeddings, distances)

    # Partial initialize
    concept = TrainedConcept(;
        index_id = index.id,
        concept = "none",
        source_doc_ids = [1, 2, 3, 4])
    @test isnothing(concept.embeddings)
    @test isnothing(concept.docs)
    @test isnothing(concept.coeffs)

    # Show default
    io = IOBuffer()
    show(io, concept)
    output = String(take!(io))
    @test output == "TrainedConcept(Concept: \"none\", Docs: -, Embeddings: -, Coeffs: -)"

    # Show full
    concept.embeddings = embeddings
    concept.docs = docs
    concept.coeffs = rand(Float32, 4)
    io = IOBuffer()
    show(io, concept)
    output = String(take!(io))
    @test output == "TrainedConcept(Concept: \"none\", Docs: 4, Embeddings: OK, Coeffs: OK)"
end

@testset "TrainedSpectrum" begin
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))
    index = DocIndex(; docs, embeddings, distances)

    # Partial initialize
    spectrum = TrainedSpectrum(;
        index_id = index.id,
        spectrum = ("none", "none2"),
        source_doc_ids = [1, 2, 3, 4])
    @test isnothing(spectrum.embeddings)
    @test isnothing(spectrum.docs)
    @test isnothing(spectrum.coeffs)

    # Show default
    io = IOBuffer()
    show(io, spectrum)
    output = String(take!(io))
    @test output ==
          "TrainedSpectrum(Spectrum: \"none\" vs. \"none2\", Docs: -, Embeddings: -, Coeffs: -)"

    # Show full
    spectrum.embeddings = embeddings
    spectrum.docs = docs
    spectrum.coeffs = rand(Float32, 4)
    io = IOBuffer()
    show(io, spectrum)
    output = String(take!(io))
    @test output ==
          "TrainedSpectrum(Spectrum: \"none\" vs. \"none2\", Docs: 4, Embeddings: OK, Coeffs: OK)"
end
