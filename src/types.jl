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
- `label::AbstractString`: Human-readable label of the topic.
- `summary::AbstractString`: Brief summary of the topic.
- `docs_idx::Vector{Int}`: Indices of documents belonging to this topic.
- `center_doc_idx::Int`: Index of the central document in this topic.
- `samples_doc_idx::Vector{Int}`: Indices of representative documents.
- `keywords_idx::Vector{Int}`: Indices of specific keywords associated with this topic.

# Example
```julia
metadata = TopicMetadata(topic_level=1, topic_idx=5)
println(metadata)
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
- `embeddings::Matrix{Float32}`: Embeddings of the documents.
- `distances::Matrix{Float32}`: Pairwise distances between document embeddings.
- `keywords_ids::T2`: Sparse matrix representing keywords in documents.
- `keywords_vocab::Vector{<:AbstractString}`: Vocabulary of keywords.
- `plot_data::Union{Nothing, Matrix{Float32}}`: 2D embedding data for plotting.
- `clustering::Any`: Results of clustering the documents.
- `topic_levels::Dict{Int, Vector{TopicMetadata}}`: Metadata for topics at different levels.

# Example
```julia
docs = ["Document 1 text", "Document 2 text"]
index = DocIndex(docs=docs, embeddings=rand(Float32, (10, 2)), distances=rand(Float32, (2, 2)))
println(index)
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
