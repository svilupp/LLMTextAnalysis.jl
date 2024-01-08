"""
    build_keywords(docs::Vector{<:AbstractString},
        return_type::Type = String;
        min_length::Int = 2,
        stopwords::Vector{String} = stopwords(Languages.English()),
        stemmer_language::Union{Nothing, String} = "english")

Extracts and returns keywords from a collection of documents.

# Arguments
- `docs`: Collection of documents from which to extract keywords. If you have only one large document, consider splitting it into smaller chunks with `PromptingTools.split_by_length`.
- `return_type`: Element type of the returned keywords. Defaults to String.
- `min_length`: Minimum length of keywords to consider. Will be dropped if they are shorter than this.
- `stopwords`: List of stopwords to exclude from keyword extraction. Defaults to English stopwords (`stopwords(Languages.English())`).
- `stemmer_language`: Language for stemming, if applicable. Set to `nothing` to disable stemming.

# Returns
- A sparse matrix where each column represents a document and each row a keyword, weighted by its frequency.
- A vector of unique keywords across all documents.

# Example
```julia
docs = ["Sample document text.", "Another document."]
keywords_ids, keywords_vocab = build_keywords(docs)
```
"""
function build_keywords(docs::Vector{<:AbstractString},
        return_type::Type = String;
        min_length::Int = 2,
        stopwords::Vector{String} = stopwords(Languages.English()),
        stemmer_language::Union{Nothing, String} = "english")
    @assert !isempty(docs) "No documents provided!"
    # Choose a stemmer
    stmr = if !isnothing(stemmer_language)
        Stemmer("english")
    else
        NoStemmer()
    end
    keywords = Vector{Vector{return_type}}(undef, length(docs))
    broadcast_lowercase(x) = lowercase.(x)
    Threads.@threads for i in eachindex(docs)
        tokens = tokenize.(split_sentences(docs[i]))
        tokens = mapreduce(broadcast_lowercase, vcat, tokens)
        filter!(t -> length(t) >= min_length && t ∉ stopwords, tokens)
        keywords[i] = tokens
    end
    ## apply stemmer
    keywords = map(x -> Snowball.stem.(Ref(stmr), x), keywords)
    keywords_vocab = unique(vcat(keywords...)) |> sort
    keywords_dict = Dict(keywords_vocab .=> 1:length(keywords_vocab))

    # Build sparse matrix of keywords
    # i-row position representing the vocab index
    # j-col position representing the document index
    count_items = length.(keywords) |> sum
    I = Vector{Int}(undef, count_items)
    J = Vector{Int}(undef, count_items)
    V = Vector{Float32}(undef, count_items)
    position = 1
    for j in eachindex(keywords)
        weight = 1 / length(keywords[j])
        for k in keywords[j]
            I[position] = keywords_dict[k] # which keyword
            J[position] = j # which document
            V[position] = weight
            position += 1
        end
    end
    keywords_ids = sparse(I, J, V, length(keywords_vocab), length(docs), +)
    return keywords_ids, keywords_vocab
end

"""
    build_index(docs::Vector{<:AbstractString}; verbose::Bool = true,
    index_id::Symbol = gensym("DocIndex"), aiembed_kwargs::NamedTuple = NamedTuple(),
    keyword_kwargs::NamedTuple = NamedTuple(), kwargs...)

Builds an index of the given documents, including their embeddings and extracted keywords.

# Arguments
- `docs`: Collection of documents to index. If you have only one large document, consider splitting it into smaller chunks with `PromptingTools.split_by_length`.
- `verbose`: Flag to enable INFO logging.
- `index_id`: Identifier for the document index. Useful if there will be multiple indices.
- `aiembed_kwargs`: Additional arguments for `PromptingTools.aiembed`. See `?aiembed` for more details.
- `keyword_kwargs`: Additional arguments for keyword extraction. See `?build_keywords` for more details.

# Returns
- An instance of `DocIndex` containing information about the documents, embeddings, keywords, etc.

# Example
```julia
docs = ["First document text", "Second document text"]
index = build_index(docs)
```
"""
function build_index(docs::Vector{<:AbstractString}; verbose::Bool = true,
        index_id::Symbol = gensym("DocIndex"), aiembed_kwargs::NamedTuple = NamedTuple(),
        keyword_kwargs::NamedTuple = NamedTuple(), kwargs...)
    @assert !isempty(docs) "No documents provided!"

    verbose && @info "Embedding $(length(docs)) documents..."
    cost_tracker = Threads.Atomic{Float64}(0.0)
    model = hasproperty(aiembed_kwargs, :model) ? aiembed_kwargs.model : PT.MODEL_EMBEDDING
    # Notice that we embed multiple docs at once, not one by one
    # OpenAI supports embedding multiple documents to reduce the number of API calls/network latency time
    # We do batch them just in case the documents are too large (targeting at most 80K characters per call)
    avg_length = sum(length.(docs)) / length(docs)
    embedding_batch_size = floor(Int, 80_000 / avg_length)
    embeddings = asyncmap(Iterators.partition(docs, embedding_batch_size)) do docs_chunk
        msg = aiembed(docs_chunk,
            normalize;
            verbose = false,
            aiembed_kwargs...)
        Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
        msg.content
    end
    embeddings = hcat(embeddings...) .|> Float32 # flatten, columns are documents
    verbose && @info "Done embedding. Total cost: \$$(round(cost_tracker[],digits=3))"
    verbose && @info "Computing pairwise distances..."
    distances = pairwise(CosineDist(), embeddings)
    verbose && @info "Extracting keywords..."
    keywords_ids, keywords_vocab = build_keywords(docs; keyword_kwargs...)
    return DocIndex(;
        id = index_id,
        docs,
        embeddings,
        distances,
        keywords_ids,
        keywords_vocab)
end

## Prepare for plotting
# TODO: allow some quick hacks to scale when there are >1M documents (eg, avoid UMAP because it's the slowest step)
"""
    prepare_plot!(index::AbstractDocumentIndex; verbose::Bool=true, kwargs...) -> AbstractDocumentIndex

Prepares the 2D UMAP plot data for a given document index.

# Arguments
- `index`: The document index to prepare plot data for.
- `verbose`: Flag to enable INFO logging.

# Returns
- The updated index with `plot_data` field populated.

# Example
```julia
index = build_index(["Some text", "More text"])
prepared_index = prepare_plot!(index)
```
"""
function prepare_plot!(index::AbstractDocumentIndex; verbose::Bool = true, kwargs...)
    if isnothing(index.plot_data)
        verbose && @info "Computing UMAP..."
        ## TODO: allow user to pass their own metric and then run on embeddings directly
        @assert :metric∉keys(kwargs) "Cannot change the metric for UMAP! We're using `:precomputed`."
        index.plot_data = umap(index.distances, 2; metric = :precomputed, kwargs...)
    end
    return index
end
