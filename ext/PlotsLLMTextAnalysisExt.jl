module PlotsLLMTextAnalysisExt

using Plots
using LLMTextAnalysis
using LLMTextAnalysis: AbstractDocumentIndex

"""
    StatsPlots.plot(index::AbstractDocumentIndex; verbose::Bool=true, k::Union{Int,Nothing}=nothing, h::Union{Float64,Nothing}=nothing, add_hover::Bool=true, cluster_kwargs::NamedTuple=NamedTuple(), labeler_kwargs::NamedTuple=NamedTuple(), plot_kwargs...) -> Plots.Plot

Generates a scatter plot of the document embeddings, colored by topic assignments.

# Arguments
- `index`: The document index.
- Additional arguments to configure the plot.

# Returns
- A scatter plot of the document embeddings.

# Example
```julia
index = build_index(["Document 1", "Document 2"])
prepare_plot!(index)
pl = StatsPlots.plot(index)
```
"""
function Plots.plot(index::AbstractDocumentIndex; verbose::Bool = true,
        k::Union{Int, Nothing} = nothing, h::Union{Float64, Nothing} = nothing,
        add_hover::Bool = true, cluster_kwargs::NamedTuple = NamedTuple(),
        labeler_kwargs::NamedTuple = NamedTuple(), plot_kwargs...)
    ## prepare plot
    prepare_plot!(index; verbose)
    ## Prepare a clustering
    previous_topic_levels = keys(index.topic_levels)
    build_clusters!(index; verbose, k, h, labeler_kwargs, cluster_kwargs...)
    ## Pick topic_level if not provided
    topic_level = if !isnothing(k)
        k
    else
        ## we don't know the exact `k`, so let pick the highest new one
        new_topics = setdiff(keys(index.topic_levels), previous_topic_levels)
        if isempty(new_topics)
            ## no new topics, so pick the highest level
            maximum(keys(index.topic_levels))
        else
            maximum(new_topics)
        end
    end
    ## Plot
    (; plot_data) = index
    topics = index.topic_levels[topic_level]
    pl = Plots.scatter(;
        size = (800, 500),
        title = "Document Embeddings",
        xlabel = "UMAP 1",
        ylabel = "UMAP 2")
    for topic in topics
        docs_idx = topic.docs_idx
        hover = if add_hover
            ["""
Topic: $(topic.label)<br>
Text: $(index.docs[doc])<br>
"""
             for doc in docs_idx]
        else
            nothing
        end
        Plots.scatter!(plot_data[1, docs_idx],
            plot_data[2, docs_idx];
            hover,
            label = topic.label,
            legend = :outertopright)
    end
    pl = plot(pl; plot_kwargs...)
    return pl
end

end # end module