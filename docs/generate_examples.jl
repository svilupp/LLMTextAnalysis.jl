using Literate

## ! Config
# example_files = joinpath(@__DIR__, "..", "examples") |> x -> readdir(x; join = true)
example_files = [
    # joinpath(@__DIR__,
    #     "..",
    #     "examples",
    #     "1_topics_in_city_of_austin_community_survey.jl")
    joinpath(@__DIR__,
        "..",
        "examples",
        "2_concept_labeling_in_city_of_austin_community_survey.jl"),
]
output_dir = joinpath(@__DIR__, "src", "examples")

# Hack to show PlotlyJS instead
function update_plotly_for_documenter(content)
    content = replace(content, "using Plots" => "import PlotlyJS, PlotlyDocumenter")
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
