using Literate

## ! Config
example_files = joinpath(@__DIR__, "..", "examples") |> x -> readdir(x; join = true)
output_dir = joinpath(@__DIR__, "src", "examples")

# Hack to show PlotlyJS
function update_plotly_for_documenter(content)
    content = replace(content, "##PLOTLYJS##" => "import PlotlyJS, PlotlyDocumenter")
    # content = replace(content, r"^\s*pl$"m => "PlotlyDocumenter.to_documenter(pl) #hide")
    content = replace(content,
        r"^\s*pl$"m => """
let k = maximum(keys(index.topic_levels))   #hide
    traces = [PlotlyJS.scatter(;    #hide
        x = index.plot_data[1, topic.docs_idx], #hide
        y = index.plot_data[2, topic.docs_idx], #hide
        name = topic.label, #hide
        text = index.docs[topic.docs_idx],  #hide
        hovertemplate = "Topic: \$(topic.label)<br>Text: %{text}<extra></extra>",    #hide
        mode = "markers")   #hide
              for topic in index.topic_levels[k]]   #hide
    layout = PlotlyJS.Layout(;   #hide
        title = "City of Austin Community Survey Themes",   #hide
        xaxis_title = "UMAP 1", #hide
        yaxis_title = "UMAP 2", template = "plotly_white")  #hide
    PlotlyJS.plot(traces, layout)|>PlotlyDocumenter.to_documenter    #hide
end #hide
""")
    content = replace(content,
        r"^\s*pl4$"m => """
let k = 4   #hide
    traces = [PlotlyJS.scatter(;    #hide
        x = index.plot_data[1, topic.docs_idx], #hide
        y = index.plot_data[2, topic.docs_idx], #hide
        name = topic.label, #hide
        text = index.docs[topic.docs_idx],  #hide
        hovertemplate = "Topic: \$(topic.label)<br>Text: %{text}<extra></extra>",    #hide
        mode = "markers")   #hide
              for topic in index.topic_levels[k]]   #hide
    layout = PlotlyJS.Layout(;   #hide
        title = "Document Embeddings",  #hide
        xaxis_title = "UMAP 1", #hide
        yaxis_title = "UMAP 2", template = "plotly_white")  #hide
    PlotlyJS.plot(traces, layout)|>PlotlyDocumenter.to_documenter    #hide
end #hide
""")

    return content
end

# Run the production loop
filter!(endswith(".jl"), example_files)
for fn in example_files
    Literate.markdown(fn,
        output_dir;
        execute = true,
        preprocess = update_plotly_for_documenter)
end

# TODO: change meta fields at the top of each file!