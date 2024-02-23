module PlotsLLMTextAnalysisExt

using Plots
using LLMTextAnalysis, Tables
using LLMTextAnalysis: AbstractDocumentIndex, label, wrap_string

"""
    Plots.plot(index::AbstractDocumentIndex; verbose::Bool = true,
        k::Union{Int, Nothing} = nothing, h::Union{Float64, Nothing} = nothing,
        text_width::Int = 30,
        add_hover::Bool = true,  hoverdata = nothing, cluster_kwargs::NamedTuple = NamedTuple(),
        labeler_kwargs::NamedTuple = NamedTuple(), plot_kwargs...)

Generates a scatter plot of the document embeddings, colored by topic assignments.

# Arguments
- `index`: The document index.
- `k`: The number of clusters to build. If not provided, the highest new level of clustering will be used.
- `h`: The height at which to cut the dendrogram. Defaults to nothing.
- `text_width`: The width of the text in the hover tooltip. If the document exceeds this width, it will be wrapped on new lines.
- `add_hover`: Whether to add a hover tooltip to the plot.
- `hoverdata`: A Tables.jl-compatible object (eg, DataFrame) with the hover data to add to each tooltip.
  Assumes that rows correspond to the individual documents in `index.docs`. Defaults to nothing.
- Additional arguments to configure the plot.

# Returns
- A scatter plot of the document embeddings.

# Example
```julia
index = build_index(["Document 1", "Document 2"])
prepare_plot!(index)
pl = plot(index)
```
"""
function Plots.plot(index::AbstractDocumentIndex; verbose::Bool = true,
        k::Union{Int, Nothing} = nothing, h::Union{Float64, Nothing} = nothing,
        text_width::Int = 30,
        add_hover::Bool = true, hoverdata = nothing,
        cluster_kwargs::NamedTuple = NamedTuple(),
        labeler_kwargs::NamedTuple = NamedTuple(), plot_kwargs...)
    @assert isnothing(hoverdata)||(Tables.istable(hoverdata) &&
                                   Tables.rowaccess(hoverdata)) "`hoverdata` must be a table and provide row access."
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
            extras = if isnothing(hoverdata)
                fill("", length(docs_idx))
            else
                subset = Tables.subset(hoverdata, docs_idx; viewhint = true)
                map(Tables.rows(subset)) do row
                    join(["<b>$(col)</b>: $(Tables.getcolumn(row,col))"
                          for col in Tables.columnnames(row)], "<br>")
                end
            end
            ["""
<b>Topic</b>: $(topic.label)<br>
<b>Text</b>: $(wrap_string(index.docs[docs_idx[i]], text_width; newline="<br>"))<br>
$(extras[i])
"""
             for i in eachindex(docs_idx, extras)]
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

"""
    Plots.plot(index::AbstractDocumentIndex,
        concept1::Union{TrainedConcept, TrainedSpectrum},
        concept2::Union{TrainedConcept, TrainedSpectrum}; verbose::Bool = true,
        text_width::Int = 30,
        add_hover::Bool = true, hoverdata = nothing,
        plot_kwargs...)


Generates a scatter plot of two concepts/spectra (see `TrainedConcept` and `TrainedSpectrum`).
It positions the documents in the index according to their scores for the two concepts/spectra.

# Arguments
- `index`: The document index.
- `concept1`: The first concept/spectrum (x-axis position).
- `concept2`: The second concept/spectrum (y-axis position).
- `text_width`: The width of the text in the hover tooltip. If the document exceeds this width, it will be wrapped on new lines.
- `add_hover`: Whether to add a hover tooltip to the plot.
- `hoverdata`: A Tables.jl-compatible object (eg, DataFrame) with the hover data to add to each tooltip.
  Assumes that rows correspond to the individual documents in `index.docs`. Defaults to nothing.
- Additional arguments to configure the plot.

# Returns
- A scatter plot of the document embeddings.

# Example

```julia
hoverdata = DataFrame(;
    extra = fill("some data", length(scores1)),
    extra2 = fill("some other data", length(scores1)))

plot(index, spectrum, concept; title = "My title", hoverdata)
```
"""
function Plots.plot(index::AbstractDocumentIndex,
        concept1::Union{TrainedConcept, TrainedSpectrum},
        concept2::Union{TrainedConcept, TrainedSpectrum};
        verbose::Bool = true,
        text_width::Int = 30,
        add_hover::Bool = true, hoverdata = nothing,
        plot_kwargs...)
    @assert isnothing(hoverdata)||(Tables.istable(hoverdata) &&
                                   Tables.rowaccess(hoverdata)) "`hoverdata` must be a table and provide row access."

    ## Plot
    scores1 = score(index, concept1)
    scores2 = score(index, concept2)

    hover = if add_hover
        extras = if isnothing(hoverdata)
            fill("", length(scores1))
        else
            map(Tables.rows(hoverdata)) do row
                join(["<b>$(col)</b>: $(Tables.getcolumn(row,col))"
                      for col in Tables.columnnames(row)], "<br>")
            end
        end
        ["""
    <b>Text</b>: \"$(wrap_string(index.docs[i],text_width; newline="<br>"))\"<br>
    <b>Score #1</b>: $(round(Int,100*scores1[i]))%<br>
    <b>Score #2</b>: $(round(Int,100*scores2[i]))%<br>
    $(extras[i])
    """
         for i in eachindex(index.docs, extras)]
    else
        nothing
    end
    pl = Plots.scatter(scores1, scores2;
        size = (800, 500),
        title = "Document Embeddings",
        label = "",
        xlabel = label(concept1),
        ylabel = label(concept2),
        yformatter = x -> "$(round(Int, 100x))%",
        xformatter = x -> "$(round(Int, 100x))%",
        hover,
        plot_kwargs...)
    return pl
end

"Creates a scatter plot of the topics behind documents in `index`. See `?plot` for details."
Plots.scatter(index::AbstractDocumentIndex; kwargs...) = Plots.plot(index; kwargs...)

"Creates a scatter plot of `concept1` vs `concept2` for the documents in `index`. See `?plot` for details."
function Plots.scatter(index::AbstractDocumentIndex,
        concept1::Union{TrainedConcept, TrainedSpectrum},
        concept2::Union{TrainedConcept, TrainedSpectrum}; kwargs...)
    Plots.plot(index, concept1, concept2; kwargs...)
end

end # end module
