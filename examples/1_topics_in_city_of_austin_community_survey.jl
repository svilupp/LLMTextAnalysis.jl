# # Example 1: Topics in the City of Austin Community Survey
# For this tutorial, we will use the [City of Austin's Community Survey](https://data.austintexas.gov/Health-and-Community-Services/2019-City-of-Austin-Community-Survey/s2py-ceb7).
# We will pick one open-ended question and extract the main themes from the answers.

# Necessary imports
using Downloads, CSV, DataFrames
using Plots
using LLMTextAnalysis
##PLOTLYJS##
plotlyjs(); # recommended backend for interactivity, install with `using Pkg; Pkg.add("PlotlyJS")`

# ## Prepare the data
# Download the survey data
Downloads.download(
    "https://data.austintexas.gov/api/views/s2py-ceb7/rows.csv?accessType=DOWNLOAD",
    joinpath(@__DIR__, "cityofaustin.csv"));

# Read the survey data into a DataFrame
df = CSV.read(joinpath(@__DIR__, "cityofaustin.csv"), DataFrame);

# Let's select one of the open-ended questions, eg,
col = "Q25 - If there was one thing you could share with the Mayor regarding the City of Austin (any comment, suggestion, etc.), what would it be?"
docs = df[!, col] |> skipmissing |> collect;

# ## Topic Analysis
# Index the documents (ie, embed them)
index = build_index(docs)

# Plot the index
#
# - You use any keywords that you're used to from Plots.jl, eg, to customize the `title` or `size`
# - `labeler_kwargs` allows us to control the LLM labeling of topics, I like the latest GPT-3.5-Turbo-1106 for the labeling. We can use any kwargs from PromptingTools.jl
# - You can specify the number of topics to show with `k`, or the height of the dendrogram to cut at with `h` (see `?Clustering.hclust`)
# - See the detail with `?plot`
pl = plot(index;
    title = "City of Austin Community Survey Themes",
    labeler_kwargs = (; model = "gpt3t"))
pl

# Voila! We have an interactive explorer of the main themes in the survey in less than 2 minutes and for a few cents!

# If you do not want to create any plots, simply call `build_clusters!(index; k)` and explore the generated topics in `index.topic_levels[k]` where `k` is the number of topics.

# ## Tip 1: Zoom in/out on the Information
#
# One of the biggest superpowers of LLMs, is that you can zoom in/out in the abstraction level to help you digest information more gradually.
# For example, we can start by looking at the top-level themes with `k=4`:
pl4 = plot(index; k = 4, labeler_kwargs = (; model = "gpt3t"))
pl4

# Now, we have both the top-level themes and the sub-themes available in `index.topic_levels`, so we can easily switch between them.

# ## Tip 2: Serialize your Index
# We don't want to recompute the index and topics every time we want to explore it, so we can serialize it to disk and load it back later.
# 
# ```julia
# using Serialization
# serialize("austin-index.jls", index)
# index = deserialize("austin-index.jls")
# ```
#
# ## Tip 3: Take Advantage of the Interactivity in PlotlyJS
#
# Remember that with PlotlyJS backend, you can zoom in/out, pan, and hover over the points to see the document text.
# Also, by single-clicking / double-clicking on the topics in the legend, you can hide/show the topics.
#
# Note: You can save the plot as an HTML file and share it with others while keeping the interactivity.
