module PlotlyJSLLMTextAnalysisExt

using PlotlyJS
using LLMTextAnalysis, Tables
using LLMTextAnalysis: AbstractDocumentIndex, label, wrap_string

"""
    PlotlyJS.plot(index::AbstractDocumentIndex; verbose::Bool = true,
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
- Additional keyword arguments to configure the layout of the plot (passed to `PlotlyJS.Layout`).

# Returns
- A scatter plot of the document embeddings.

# Example
```julia
index = build_index(["Document 1", "Document 2"])
prepare_plot!(index)
pl = PlotlyJS.plot(index; title = "MyPlot")
```
"""
function PlotlyJS.plot(index::AbstractDocumentIndex; verbose::Bool = true,
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

    traces = PlotlyJS.AbstractTrace[]
    for topic in topics
        docs_idx = topic.docs_idx
        if add_hover
            if isnothing(hoverdata)
                hovertemplate = "<b>Topic:</b> $(topic.label)<br><b>Text:</b> %{text}<extra></extra>"
                customdata = fill("", length(docs_idx))

            else
                hovertemplate = "<b>Topic:</b> $(topic.label)<br><b>Text:</b> %{text}<br>%{customdata}<extra></extra>"
                subset = Tables.subset(hoverdata, docs_idx; viewhint = true)
                customdata = map(Tables.rows(subset)) do row
                    join(["<b>$(col)</b>: $(Tables.getcolumn(row,col))"
                          for col in Tables.columnnames(row)], "<br>")
                end
            end
        else
            hovertemplate = ""
        end
        trace = PlotlyJS.scatter(;
            x = plot_data[1, docs_idx],
            y = plot_data[2, docs_idx],
            name = topic.label,
            text = wrap_string.(index.docs[topic.docs_idx], text_width; newline = "<br>"),
            hovertemplate,
            customdata,
            mode = "markers")
        push!(traces, trace)
    end

    layout = PlotlyJS.Layout(;
        title = "Document Embeddings",
        xaxis_title = "UMAP 1",
        yaxis_title = "UMAP 2", template = "plotly_white",
        plot_kwargs...)
    pl = PlotlyJS.plot(traces, layout)

    return pl
end

"""
    PlotlyJS.plot(index::AbstractDocumentIndex,
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
- Additional keyword arguments to configure the layout of the plot (passed to `PlotlyJS.Layout`).

# Returns
- A scatter plot of the document embeddings.

# Example

```julia
# assume we already have index, spectrum, concept

hoverdata = DataFrame(;
    extra = fill("some data", length(index.docs)),
    extra2 = fill("some other data", length(index.docs)))
pl = PlotlyJS.plot(index, spectrum, concept; title = "MyTitle", hoverdata)
```
"""
function PlotlyJS.plot(index::AbstractDocumentIndex,
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

    if add_hover
        if isnothing(hoverdata)
            hovertemplate = "<b>Text:</b> %{text}<br><b>Score #1</b>: %{x}<br><b>Score #2</b>: %{y}<extra></extra>"
            customdata = fill("", length(scores1))

        else
            hovertemplate = "<b>Text:</b> %{text}<br><b>Score #1</b>: %{x}<br><b>Score #2</b>: %{y:,.0%}<br>%{customdata}<extra></extra>"
            customdata = map(Tables.rows(hoverdata)) do row
                join(["<b>$(col)</b>: $(Tables.getcolumn(row,col))"
                      for col in Tables.columnnames(row)], "<br>")
            end
        end
    else
        hovertemplate = ""
    end

    trace = PlotlyJS.scatter(; x = scores1, y = scores2,
        text = wrap_string.(index.docs, text_width; newline = "<br>"),
        hovertemplate,
        customdata,
        mode = "markers")

    layout = PlotlyJS.Layout(;
        title = "Document Embeddings",
        xaxis_title = label(concept1),
        yaxis_title = label(concept2),
        xaxis_tickformat = ",.0%",
        yaxis_tickformat = ",.0%",
        template = "plotly_white",
        plot_kwargs...)
    pl = PlotlyJS.plot(trace, layout)

    return pl
end

"Creates a scatter plot of the topics behind documents in `index`. See `?plot` for details."
PlotlyJS.scatter(index::AbstractDocumentIndex; kwargs...) = PlotlyJS.plot(index; kwargs...)

"Creates a scatter plot of `concept1` vs `concept2` for the documents in `index`. See `?plot` for details."
function PlotlyJS.scatter(index::AbstractDocumentIndex,
        concept1::Union{TrainedConcept, TrainedSpectrum},
        concept2::Union{TrainedConcept, TrainedSpectrum}; kwargs...)
    PlotlyJS.plot(index, concept1, concept2; kwargs...)
end

end # end module
