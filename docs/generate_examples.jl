using Literate

## ! Config
# example_files = joinpath(@__DIR__, "..", "examples") |> x -> readdir(x; join = true)
example_files = [
# joinpath(@__DIR__,
#     "..",
#     "examples",
#     "1_topics_in_city_of_austin_community_survey.jl")
## joinpath(@__DIR__,
##     "..",
##     "examples",
##     "3_customize_topic_labels.jl"),
    joinpath(@__DIR__,
    "..",
    "examples",
    "4_classify_documents.jl")
]
output_dir = joinpath(@__DIR__, "src", "examples")

# Hack to show PlotlyJS instead
function update_plotly_for_documenter(content)
    content = replace(content,
        "using Plots" => "import PlotlyJS, PlotlyDocumenter  ## Only for the documentation, not needed for users!")
    content = replace(content, r"^\s*plotlyjs().*?$"m => "")
    content = replace(content, "plot(" => "PlotlyJS.plot(")
    content = replace(content,
        r"^\s*(pl\d{0,1})$"m => s"PlotlyDocumenter.to_documenter(\1) #hide")
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
