# # Example 2: Label Arbitrary Concepts in the City of Austin Community Survey
# For this tutorial, we will use the [City of Austin's Community Survey](https://data.austintexas.gov/Health-and-Community-Services/2019-City-of-Austin-Community-Survey/s2py-ceb7).
#
# We will pick one open-ended question. Let's say we want to help the mayor to prioritize ideas,
# so we will lay out the verbatims against the concepts of being "action-oriented" and "forward-looking".
#
# You can choose any concepts that you want.

# Necessary imports
using Downloads, CSV, DataFrames
using Plots
using LLMTextAnalysis
plotlyjs(); # plotlyjs() is the recommended backend for Plots.jl for interactivity, install with `using Pkg; Pkg.add("PlotlyJS")`

# ## Prepare the Data
# Download the survey data
Downloads.download("https://data.austintexas.gov/api/views/s2py-ceb7/rows.csv?accessType=DOWNLOAD",
    joinpath(@__DIR__, "cityofaustin.csv"));

# Read the survey data into a DataFrame
df = CSV.read(joinpath(@__DIR__, "cityofaustin.csv"), DataFrame);

# Let's select one of the open-ended questions, eg,
col = "Q25 - If there was one thing you could share with the Mayor regarding the City of Austin (any comment, suggestion, etc.), what would it be?"
docs = df[!, col] |> skipmissing |> collect;

# ## Build the Index
# Index the documents (ie, embed them)
index = build_index(docs)

# Sometimes you know what you're looking for, but it's hard to define the exact keywords. For example, you might want to identify documents that are "action-oriented" or "pessimistic" or "forward-looking".
#
# For these situations, `LLMTextAnalysis` offers two distinct functions for document analysis: `train_concept` and `train_spectrum`. Each serves a different purpose in text analysis:
#
# - **`train_concept`**: Focuses on analyzing a single, specific concept within documents (eg, "action-oriented")
# - **`train_spectrum`**: Analyzes documents in the context of two opposing concepts (eg, "optimistic" vs. "pessimistic" or "forward-looking" vs. "backward-looking")
#
# The resulting return types are `TrainedConcept` and `TrainedSpectrum`, respectively. Both can be used to score documents on the presence of the concept or their position on the spectrum.
#
# Why do we need `train_spectrum` and not simply use two `TrainedConcepts`? It's because opposite of "forward-looking" can be many things, eg, "short-sighted", "dwelling in the past", or simply "not-forward looking". 
#
# `train_spectrum` allows you to define the opposite concept that you need and score documents on the spectrum between the two.

# ## Score Documents against a Concept

# Let's say we want to identify documents that are "action-oriented". 
# We can use `train_concept` to train a model to identify documents that are "action-oriented" and score the documents against the concept.
#
# Let's show the top 5 documents that are most "action-oriented".

concept = train_concept(index,
    "action-oriented";
    aigenerate_kwargs = (; model = "gpt3t"))

scores = score(index, concept)
index.docs[first(sortperm(scores, rev = true), 5)]

# ## Score Documents along a Spectrum

# We may want to define an arbitrary "spectrum" (axis/polar opposites) and score documents on it.
# Let's introduce a spectrum for "dwelling in the past" vs "forward-looking". 
# The higher the score (eg, 100%), the more "forward-looking" the document/text is.
#
# Let's show the top 5 documents that are most "forward-looking".

spectrum = train_spectrum(index,
    ("dwelling in the past", "forward-looking");
    aigenerate_kwargs = (; model = "gpt3t"))

scores = score(index, spectrum)
index.docs[first(sortperm(scores, rev = true), 5)]

# And how about the ones "dwelling in the past" (set `rev=false`)?
index.docs[first(sortperm(scores, rev = false), 5)]

# ## Summarize via Plot

# Let's interactively explore our results.
#
# We can use `plot` to plot the documents along the trained concepts and spectrums (simple scatter plot).
# The positions of args `concept` and `spectrum` are important, as they determine the position of the concepts in the plot (x-axis, y-axis)

pl = plot(index, spectrum, concept;
    title = "Prioritizing Action-Oriented and Forward-Looking Ideas (Top-right Corner)")
pl

#
# What if you need to add some additional information to the tooltip for each data point? 
# You can do that with `hoverdata` argument, see `?plot` for more details.