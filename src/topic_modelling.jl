"""
    build_topic(index::AbstractDocumentIndex, assignments::Vector{Int}, topic_idx::Int;
    topic_level::Int = nunique(assignments),
    verbose::Bool = false, add_label::Bool = true, add_summary::Bool = false,
    label_template::Union{Nothing, Symbol} = :TopicLabelerBasic,
    summary_template::Union{Nothing, Symbol} = :TopicSummarizerBasic,
    num_samples::Int = 8, num_keywords::Int = 10,
    cost_tracker::Union{Nothing, Threads.Atomic{Float64}} = nothing, aikwargs...)

Builds the metadata for a specific topic in the document index.

# Arguments
- `index`: The document index.
- `assignments`: Vector of topic assignments for each document.
- `topic_idx`: Index of the topic to build metadata for.

# Keyword Arguments
- `topic_level`: The level of the topic in the hierarchy. Corresponds to `k` in `build_clusters!`.
- `verbose`: Flag to enable INFO logging.
- `add_label`: Flag to enable topic labeling, ie, call LLM to generate topic label.
- `add_summary`: Flag to enable topic summarization, ie, call LLM to generate topic summary.
- `label_template`: The LLM template to use for topic labeling. See `?aitemplates` for more details on templates.
- `summary_template`: The LLM template to use for topic summarization. See `?aitemplates` for more details on templates.
- `num_samples`: Number of diverse samples to show to the LLM for each topic.
- `num_keywords`: Number of top keywords to show to the LLM for each topic.
- `cost_tracker`: An `Atomic` to track the cost of the LLM calls, if we trigger multiple calls asynchronously.


# Returns
- `TopicMetadata` instance for the specified topic.

# Example
```julia
index = build_index(["Document 1", "Document 2"])
assignments = [1, 1]
metadata = build_topic(index, assignments, 1)
```
"""
function build_topic(
        index::AbstractDocumentIndex, assignments::Vector{Int}, topic_idx::Int;
        topic_level::Int = nunique(assignments),
        verbose::Bool = false, add_label::Bool = true, add_summary::Bool = false,
        label_template::Union{Nothing, Symbol} = :TopicLabelerBasic,
        summary_template::Union{Nothing, Symbol} = :TopicSummarizerBasic,
        num_samples::Int = 8, num_keywords::Int = 10,
        cost_tracker::Union{Nothing, Threads.Atomic{Float64}} = nothing, aikwargs...)
    @assert topic_idx ∈ assignments "Topic index $topic_idx not found in assignments!"
    @assert !isnothing(label_template)||!add_label "No label template provided!"
    @assert !isnothing(summary_template)||!add_summary "No summary template provided!"
    @assert num_samples>=0 "Number of samples must be non-negative!"
    @assert num_keywords>=0 "Number of keywords must be non-negative!"

    (; docs, embeddings, distances, keywords_ids, keywords_vocab) = index

    mask = assignments .== topic_idx
    docs_idx = findall(mask)
    docs_masked = @view(docs[mask])

    # Extract top_k keywords
    if num_keywords > 0 && !isempty(keywords_ids)
        sum_weights = sum(keywords_ids, dims = 2) |> vec
        keywords_idx = first(sortperm(sum_weights, rev = true), num_keywords)
        keywords = join(@view(keywords_vocab[keywords_idx]), ", ")
    else
        keywords_idx = Int[]
        keywords = "Not provided."
    end

    # find the central document
    center = mean(@view(embeddings[:, mask]); dims = 2)
    center_doc_idx = pairwise(CosineDist(), @view(embeddings[:, mask]), center) |> argmin |>
                     x -> x[1]
    central_text = docs_masked[center_doc_idx]

    # Find a spread out sample of diverse documents
    if num_samples > 0
        num_samples_ = min(num_samples, length(docs_idx))
        samples_doc_idx = kmedoids(@view(distances[mask, mask]), num_samples_).medoids
        # Remove the center doc from the samples
        filter!(x -> x != center_doc_idx, samples_doc_idx)
        # Double check if we removed all samples
        samples = length(samples_doc_idx) > 0 ?
                  "-" * join(@view(docs_masked[samples_doc_idx]), "\n- ") : "Not provided."
    else
        samples_doc_idx = Int[]
        samples = "Not provided."
    end

    # Generate label and summary
    model = hasproperty(aikwargs, :model) ? aikwargs.model : PT.MODEL_CHAT
    label = if add_label && !isnothing(label_template)
        # TODO: check the template if it has all the valid placeholders
        msg = aigenerate(label_template;
            central_text,
            samples,
            keywords,
            verbose,
            aikwargs...)
        !isnothing(cost_tracker) &&
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
        ## quick hack for weaker models that repeat the sentence
        clean = split(msg.content, "###\n")[end]
        clean = split(clean, "topic name is:")[end]
        strip(clean)
    else
        ""
    end
    summary = if add_summary && !isnothing(summary_template)
        # TODO: check the template if it has all the valid placeholders
        msg = aigenerate(summary_template;
            central_text,
            samples,
            keywords,
            verbose,
            aikwargs...)
        !isnothing(cost_tracker) &&
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
        ## quick hack for weaker models that repeat the sentence
        clean = split(msg.content, "###\n")[end]
        strip(clean)
    else
        ""
    end

    return TopicMetadata(;
        index_id = index.id,
        topic_level,
        topic_idx,
        label,
        summary,
        docs_idx,
        center_doc_idx,
        samples_doc_idx,
        keywords_idx)
end

## Clustering

"""
    build_clusters!(index::AbstractDocumentIndex; k::Union{Int, Nothing} = nothing,
        h::Union{Float64, Nothing} = nothing,
        verbose::Bool = true, add_label::Bool = true, add_summary::Bool = false,
        labeler_kwargs::NamedTuple = NamedTuple(),
        cluster_kwargs...)

Performs clustering on the document index and builds topics at different levels.

# Arguments
- `index`: The document index.
- `k`: Number of clusters to cut at.
- `h`: Height to cut the dendrogram at. See `?Clustering.hclust` for more details.
- `verbose`: Flag to enable INFO logging.
- `add_label`: Flag to enable topic labeling, ie, call LLM to generate topic label.
- `add_summary`: Flag to enable topic summarization, ie, call LLM to generate topic summary.
- `labeler_kwargs`: Keyword arguments to pass to the LLM labeler. See `?build_topic` for more details on available arguments.
- `cluster_kwargs`: All remaining arguments will be passed to `Clustering.hclust`. See `?Clustering.hclust` for more details on available arguments.


# Returns
- The updated index with clustering information and topic metadata.

# Example
```julia
index = build_index(["Doc 1", "Doc 2"])
clustered_index = build_clusters!(index, k=2)
```
"""
function build_clusters!(index::AbstractDocumentIndex; k::Union{Int, Nothing} = nothing,
        h::Union{Float64, Nothing} = nothing,
        verbose::Bool = true, add_label::Bool = true, add_summary::Bool = false,
        labeler_kwargs::NamedTuple = NamedTuple(),
        cluster_kwargs...)
    if isnothing(index.clustering)
        verbose && @info "Building hierarchical clusters..."
        index.clustering = hclust(index.distances; linkage = :complete, cluster_kwargs...)
    end
    ## cluster assignments
    cost_tracker = Threads.Atomic{Float64}(0.0)
    topic_kwargs = (;
        verbose = false,
        cost_tracker,
        add_label,
        add_summary,
        labeler_kwargs...)
    remember_count_clusters = nothing
    if !isnothing(k)
        verbose && @info "Cutting clusters at k=$k..."
        assignments = cutree(index.clustering; k)
        count_topics = nunique(assignments)
        topics = asyncmap(
            i -> build_topic(index,
                assignments,
                i;
                topic_kwargs...,
                topic_level = count_topics),
            1:count_topics)
        verbose &&
            @info "Done building $count_topics topics. Cost: \$$(round(cost_tracker[],digits=3))"
        index.topic_levels[count_topics] = topics
        remember_count_clusters = count_topics
    end
    if !isnothing(h)
        verbose && @info "Cutting clusters at h=$h..."
        assignments = cutree(index.clustering; h)
        count_topics = nunique(assignments)
        ## early exit to not duplicate work
        count_topics == remember_count_clusters && return index
        topics = asyncmap(
            i -> build_topic(index,
                assignments,
                i;
                topic_kwargs...,
                topic_level = count_topics),
            1:count_topics)
        verbose &&
            @info "Done building $count_topics topics. Cost: \$$(round(cost_tracker[],digits=3))"
        index.topic_levels[count_topics] = topics
    end
    if isempty(index.topic_levels)
        k = ceil(Int, log(length(index.docs))) * 2
        verbose && @info "Cutting clusters at k=$k..."
        assignments = cutree(index.clustering; k)
        count_topics = nunique(assignments)
        topics = asyncmap(
            i -> build_topic(index,
                assignments,
                i;
                topic_kwargs...,
                topic_level = count_topics),
            1:count_topics)
        verbose &&
            @info "Done building $count_topics topics. Cost: \$$(round(cost_tracker[],digits=3))"
        index.topic_levels[count_topics] = topics
    end
    return index
end
