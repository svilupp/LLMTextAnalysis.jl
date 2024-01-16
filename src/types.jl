# Basic types for the package
# - NoStemmer: a dummy stemmer that does nothing
# - TopicMetadata: metadata for a topic, vector of these is stored in a document index
# - DocIndex: the main data structure for the package, holds the documents and their embeddings

abstract type AbstractTopicMetadata end
"""
    TopicMetadata <: AbstractTopicMetadata

A struct representing the metadata of a specific topic extracted from a collection of documents.

# Fields
- `index_id::Symbol`: Identifier for the topic.
- `topic_level::Int`: The level of the topic in the hierarchy.
- `topic_idx::Int`: Index of the topic.
- `label::AbstractString`: Human-readable label of the topic. Defaults to `""`.
- `summary::AbstractString`: Brief summary of the topic. Defaults to `""`.
- `docs_idx::Vector{Int}`: Indices of documents belonging to this topic. Corresponds to positions in `index.docs`.
- `center_doc_idx::Int`: Index of the central document in this topic. Corresponds to a position in `docs_idx` (not index!)
- `samples_doc_idx::Vector{Int}`: Indices of representative documents. Corresponds to positions in `docs_idx` (not index!)
- `keywords_idx::Vector{Int}`: Indices of specific keywords associated with this topic. Corresponds to positions in `index.keywords_vocab`.

# Example
```julia
topic = TopicMetadata(topic_level=1, topic_idx=5)
```
"""
@kwdef mutable struct TopicMetadata <: AbstractTopicMetadata
    index_id::Symbol
    ## eg, k=5
    topic_level::Int
    topic_idx::Int
    label::AbstractString = ""
    summary::AbstractString = ""
    ## The documents that belong to this topic
    docs_idx::Vector{Int} = Int[]
    ## The most central document, its position in `docs_idx`
    center_doc_idx::Int = -1
    ## The representative documents for the topic, their position in `docs_idx`
    samples_doc_idx::Vector{Int} = Int[]
    ## Specific keywords in `index.keywords_vocab` that are associated with this topic
    keywords_idx::Vector{Int} = Int[]
end

abstract type AbstractDocumentIndex end
"""
    DocIndex{T1<:AbstractString, T2<:AbstractMatrix} <: AbstractDocumentIndex

A struct for maintaining an index of documents, their embeddings, and related information.

# Fields
- `id::Symbol`: Unique identifier for the document index.
- `docs::Vector{T1}`: Collection of documents.
- `embeddings::Matrix{Float32}`: Embeddings of the documents. Documents are columns.
- `distances::Matrix{Float32}`: Pairwise distances between document embeddings. Documents are columns.
- `keywords_ids::T2`: Sparse matrix representing keywords in documents. Keywords in `keywords_vocab` are rows, documents are columns.
- `keywords_vocab::Vector{<:AbstractString}`: Vocabulary of keywords.
- `plot_data::Union{Nothing, Matrix{Float32}}`: 2D embedding data for plotting. Rows are dimensions, columns are documents.
- `clustering::Any`: Results of clustering the documents.
- `topic_levels::Dict{Int, Vector{TopicMetadata}}`: Metadata for topics at different levels. Indexed by `k` = number of topics.

# Example
```julia
docs = ["Document 1 text", "Document 2 text"]
index = DocIndex(docs=docs, embeddings=rand(Float32, (10, 2)), distances=rand(Float32, (2, 2)))
```
"""
@kwdef mutable struct DocIndex{T1 <: AbstractString, T2 <: AbstractMatrix} <:
                      AbstractDocumentIndex
    id::Symbol = gensym("DocIndex")
    ## documents/document chunks to be embedded
    docs::Vector{T1}
    ## embeddings: rows are dimensions, columns are documents
    embeddings::Matrix{Float32}
    distances::Matrix{Float32}
    ## keywords_ids: rows are the vocabulary, columns are documents
    keywords_ids::T2 = sparse(Float32[], Float32[], Float32[])
    keywords_vocab::Vector{<:AbstractString} = Vector{String}()
    ## two-dimensional embedding of the documents for plotting
    plot_data::Union{Nothing, Matrix{Float32}} = nothing
    ## clustering results
    clustering::Any = nothing
    ## Holds the topic metadata for each topic level (eg, k=5)
    topic_levels::Dict{Int, Vector{TopicMetadata}} = Dict{Int, Vector{TopicMetadata}}()
end

## Show methods
function Base.show(io::IO, topic::TopicMetadata)
    (; topic_idx, topic_level, label, summary, docs_idx) = topic
    label_str = isempty(label) ? "-" : "\"" * label * "\""
    summary_str = isempty(summary) ? "-" : "Available"
    print(io,
        nameof(typeof(topic)),
        "(ID: $(topic_idx)/$(topic_level), Documents: $(length(docs_idx)), Label: $(label_str),  Summary: $(summary_str))")
end

function Base.show(io::IO, index::DocIndex)
    plot_data_str = isnothing(index.plot_data) ? "None" : "OK"
    topic_levels_str = isempty(index.topic_levels) ? "None" :
                       "$(join(keys(index.topic_levels),", "))"
    print(io,
        nameof(typeof(index)),
        "(Documents: $(length(index.docs)), PlotData: $plot_data_str, Topic Levels: $topic_levels_str)")
end

## Concept Labeling
"""
    TrainedConcept

The `TrainedConcept` struct is used for representing and working with a trained concept model.

It encapsulates all the necessary information required to analyze and score documents based on their relevance to a specific concept.

# Fields

- `index_id`: A unique identifier for the `AbstractDocumentIndex` associated with this concept.
- `source_doc_ids`: Indices of the documents from the `AbstractDocumentIndex` used for training the concept model. Corresponds to `index.docs`
- `concept`: The specific concept (as a string) that this model is trained to analyze.
- `docs`: The collection of rewritten documents, which are modified to reflect the concept.
- `embeddings`: The embeddings of the rewritten documents, used for training the model. Columns are documents, rows are dimensions.
- `coeffs`: The coefficients of the trained logistic regression model. Maps to each dimension in `embeddings`.


# Example

```julia
index = build_index(...)

# Training a concept
concept = train_concept(index, "sustainability")

# Using TrainedConcept for scoring
scores = score(index, concept)
# or use it as a functor: `scores = concept(index)`

# Accessing the model details
println("Concept: ", concept.concept)
println("Coefficients: ", concept.coeffs)
println("Source Document IDs: ", concept.source_doc_ids)
println("Re-written Documents: ", concept.docs) # good for debugging if results are poor
```
"""
@kwdef mutable struct TrainedConcept
    index_id::Symbol # source index
    # required, list of source document positions in index
    source_doc_ids::Vector{Int}
    # what concept we're training
    concept::String
    # generate docs - first for spectrum1, then spectrum2, ie, 2x length(source_doc_ids)
    docs::Union{Vector{<:AbstractString}, Nothing} = nothing
    # embeddings of the generated docs: (embedding_size, num_docs)
    embeddings::Union{Matrix{Float32}, Nothing} = nothing
    coeffs::Union{Vector{Float32}, Nothing} = nothing
end

"""
    TrainedSpectrum

The `TrainedSpectrum` is used to score documents across a spectrum defined by two contrasting concepts. 

It encapsulates the essential information required to evaluate and score documents based on their alignment with the specified spectrum.

Note: `TrainedSpectrum` supports functor behavior, allowing it to be used as a function to score documents in an `AbstractDocumentIndex` based on their alignment with the spectrum.

# Fields

- `index_id`: A unique identifier for the `AbstractDocumentIndex` associated with the spectrum.
- `source_doc_ids`: Indices of the documents from the `AbstractDocumentIndex` used for training the spectrum model. Corresponds to `index.docs`.
- `spectrum`: A tuple containing the two contrasting concepts (as strings) that define the spectrum.
- `docs`: A collection of rewritten documents, modified to align with the two ends of the spectrum.
- `embeddings`: The embeddings of the rewritten documents, used for training the model. Columns are documents, rows are dimensions.
- `coeffs`: Coefficients of the trained logistic regression model, corresponding to each dimension in `embeddings`.

# Example

```julia

index = build_index(...)

# Create and train a spectrum model
spectrum = TrainedSpectrum(index, ("innovation", "tradition"))

# Using TrainedSpectrum for scoring
scores = score(index, concept)
# or use it as a functor: `scores = spectrum(index)`

# Accessing the model details
println("Spectrum: ", spectrum.spectrum)
println("Coefficients: ", spectrum.coeffs)
println("Source Document IDs: ", spectrum.source_doc_ids)
println("Re-written Documents: ", spectrum.docs) # good for debugging if results are poor
```
"""
@kwdef mutable struct TrainedSpectrum
    index_id::Symbol # source index
    # required, list of source document positions in index
    source_doc_ids::Vector{Int}
    # required, spectrum1 and spectrum2 lenses to rewrite the text in
    spectrum::Tuple{String, String}
    # generate docs - first for spectrum1, then spectrum2, ie, 2x length(source_doc_ids)
    docs::Union{Vector{<:AbstractString}, Nothing} = nothing
    # embeddings of the generated docs: (embedding_size, num_docs)
    embeddings::Union{Matrix{Float32}, Nothing} = nothing
    coeffs::Union{Vector{Float32}, Nothing} = nothing
end

function Base.show(io::IO, obj::TrainedConcept)
    (; concept, docs, embeddings, coeffs) = obj
    docs_str = isnothing(docs) ? "-" : "$(length(docs))"
    embeddings_str = isnothing(embeddings) ? "-" : "OK"
    coefficients_str = isnothing(coeffs) ? "-" : "OK"

    print(io,
        nameof(typeof(obj)),
        "(Concept: \"$(concept)\", Docs: $docs_str, Embeddings: $embeddings_str, Coeffs: $coefficients_str)")
end
function Base.show(io::IO, obj::TrainedSpectrum)
    (; spectrum, docs, embeddings, coeffs) = obj
    docs_str = isnothing(docs) ? "-" : "$(length(docs))"
    embeddings_str = isnothing(embeddings) ? "-" : "OK"
    coefficients_str = isnothing(coeffs) ? "-" : "OK"

    print(io,
        nameof(typeof(obj)),
        "(Spectrum: \"$(spectrum[1])\" vs. \"$(spectrum[2])\", Docs: $docs_str, Embeddings: $embeddings_str, Coeffs: $coefficients_str)")
end

## Other

"""
    NoStemmer

A dummy stemmer used as a workaround for bypassing stemming in keyword extraction. 

# Example
```julia
Snowball.stem(NoStemmer(), ["running", "jumps"]) # returns ["running", "jumps"]
```
"""
struct NoStemmer end
# workaround for stemming to be passthrough when necessary
Snowball.stem(::NoStemmer, words) = words
