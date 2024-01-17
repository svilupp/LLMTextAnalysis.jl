@testset "plot-index" begin
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))

    index = DocIndex(; docs, embeddings, distances)
    prepare_plot!(index; n_neighbors = 1)
    index.topic_levels[2] = [
        TopicMetadata(;
            index_id = index.id,
            topic_idx = 1,
            topic_level = 2,
            docs_idx = [1, 2],
            label = "Topic Name",
            summary = "Topic Name"),
        TopicMetadata(;
            index_id = index.id,
            topic_idx = 2,
            topic_level = 2,
            docs_idx = [3, 4],
            label = "Topic Name",
            summary = "Topic Name")]
    hoverdata = Tables.table(fill("something", length(index.docs), 2);
        header = [:extra, :extra2])
    # Plots.jl
    Plots.plot(index;
        title = "abc",
        hoverdata)

    Plots.scatter(index;
        title = "abc",
        hoverdata)
    # PlotlyJS.jl
    PlotlyJS.plot(index;
        title = "abc",
        hoverdata)
    PlotlyJS.scatter(index;
        title = "abc",
        hoverdata)
end

@testset "plot-concept" begin
    docs = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text"]
    embeddings = ones(Float32, (10, 4))
    distances = zeros(Float32, (4, 4))

    index = DocIndex(; docs, embeddings, distances)
    prepare_plot!(index; n_neighbors = 1)
    index.topic_levels[2] = [
        TopicMetadata(;
            index_id = index.id,
            topic_idx = 1,
            topic_level = 2,
            docs_idx = [1, 2],
            label = "Topic Name",
            summary = "Topic Name"),
        TopicMetadata(;
            index_id = index.id,
            topic_idx = 2,
            topic_level = 2,
            docs_idx = [3, 4],
            label = "Topic Name",
            summary = "Topic Name")]
    concept = TrainedConcept(;
        index_id = index.id, source_doc_ids = [1, 2],
        concept = "x",
        coeffs = ones(Float32, 10))
    spectrum = TrainedSpectrum(;
        index_id = index.id, source_doc_ids = [1, 2],
        spectrum = ("x", "y"), coeffs = 2ones(Float32, 10))
    hoverdata = Tables.table(fill("something", length(index.docs), 2);
        header = [:extra, :extra2])
    # Plots.jl
    Plots.plot(index, concept, spectrum;
        title = "abc",
        hoverdata)

    Plots.scatter(index, spectrum, concept;
        title = "abc",
        hoverdata)
    # PlotlyJS.jl
    PlotlyJS.plot(index, concept, spectrum;
        title = "abc",
        hoverdata)
    PlotlyJS.scatter(index, concept, spectrum;
        title = "abc",
        hoverdata)
end