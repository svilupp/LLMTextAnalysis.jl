# LLMTextAnalysis.jl: "Unveil Text Insights with LLMs" 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://svilupp.github.io/LLMTextAnalysis.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://svilupp.github.io/LLMTextAnalysis.jl/dev/) [![Build Status](https://github.com/svilupp/LLMTextAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/svilupp/LLMTextAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/svilupp/LLMTextAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/svilupp/LLMTextAnalysis.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


## Introduction
LLMTextAnalysis.jl is a Julia package crafted to harness the power of Large Language Models (LLMs) for effectively identifying and labeling topics within document corpora. It offers an efficient way to analyze textual data, providing insights into underlying themes and concepts.

### Quick Start
Jump right into LLMTextAnalysis.jl with this simple example:

Note: You need to have a working LLM setup to run this example. See [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl). It takes a maximum of two minutes to get the OpenAI API key.

```julia
using LLMTextAnalysis
texts = ["I like fries","I like burgers","How do you cook chicken?", "Recently, I read a fantasy novel about dragons"] # some vector of documents
index = build_index(texts)
```

You'll see the following output:

```plaintext
[ Info: Embedding 4 documents...
[ Info: Done embedding. Total cost: $0.0
[ Info: Computing pairwise distances...
[ Info: Extracting keywords...

DocIndex(Documents: 4, PlotData: None, Topic Levels: 2)
```

Now, you can use the `build_clusters!` function to add some topic clusters:
```julia
build_clusters!(index; k=2) # if you don't choose k, it will be chosen automatically
# review the topics generated when we ask for 2 clusters
index.topic_levels[2]
```

```plaintext
[ Info: Building hierarchical clusters...
[ Info: Cutting clusters at k=2...
[ Info: Done building 2 topics. Cost: $0.0

2-element Vector{TopicMetadata}:
 TopicMetadata(ID: 1/2, Documents: 3, Label: "Food Preferences and Cooking Techniques",  Summary: -)
 TopicMetadata(ID: 2/2, Documents: 1, Label: "Dragons in Fantasy Novels",  Summary: -)
```

Or you can just call the `plot` function (it will create some topic clusters automatically under the hood):
```julia
using Plots
plot(index)
```

For some visual examples, scroll down to the [Basic Usage](#basic-usage) section.

### Installation and Setup
Install LLMTextAnalysis.jl from the REPL with `]LLMTextAnalysis` or via:

```julia
using Pkg
Pkg.add("LLMTextAnalysis")
```
 
The package depends on PromptingTools.jl, which facilitates integration with various Large Language Models. We recommend OpenAI for its efficiency, cost-effectiveness, and privacy. See [PromptingTools.jl documentation](https://github.com/svilupp/PromptingTools.jl) for setup details.

### Explore Topics

Start analyzing your document corpus with these steps:

1. Load your documents into the package.
2. Use the `build_index` function to process your texts.
3. Use the `plot` function to visualize the results. It will call all the supporting functions under the hood.

A good starting point is the City of Austin Community Survey, available [here](https://data.austintexas.gov/dataset/Community-Survey/s2py-ceb7/data).

```julia
using Downloads, CSV, DataFrames
using Plots
using LLMTextAnalysis
plotlyjs() # recommended backend for interactivity, install with `using Pkg; Pkg.add("PlotlyJS")`

## Load the data
df = CSV.read(joinpath(@__DIR__, "cityofaustin.csv"), DataFrame);
col = "Q25 - If there was one thing you could share with the Mayor regarding the City of Austin (any comment, suggestion, etc.), what would it be?"
docs = df[!, col] |> skipmissing |> collect;

## Explore the topics in just 2 lines of code
index = build_index(docs)
pl = plot(index; title = "City of Austin Community Survey Themes")
```

![City of Austin Community Survey Themes](docs/src/assets/austin_scatter.png)


Run the full example via `examples/1_topics_in_city_of_austin_community_survey.jl`.

You can easily show the topic tree for multiple levels of k, ie, see how different topics nest into each other.

Let's ensure we have several topic levels, eg, k=4, k=10, k=20.
```julia
build_clusters!(index; k=4)
build_clusters!(index; k=10)
build_clusters!(index; k=20)
```

Let's produce the topic tree:
```julia
root = topic_tree(index, [4, 10, 20])
print_tree(root)
```



Now you know where most of the documents are!

### Identify and Score Documents on Arbitrary Concepts / Spectrum

Sometimes you know what you're looking for, but it's hard to define the exact keywords. For example, you might want to identify documents that are "action-oriented" or "pessimistic" or "forward-looking".

For these situations, `LLMTextAnalysis` offers two distinct functions for document analysis: `train_concept` and `train_spectrum`. Each serves a different purpose in text analysis:

- **`train_concept`**: Focuses on analyzing a single, specific concept within documents (eg, "action-oriented")
- **`train_spectrum`**: Analyzes documents in the context of two opposing concepts (eg, "optimistic" vs. "pessimistic" or "forward-looking" vs. "backward-looking")

The resulting return types are `TrainedConcept` and `TrainedSpectrum`, respectively. Both can be used to score documents on the presence of the concept or their position on the spectrum.

Why do we need `train_spectrum` and not simply use two `TrainedConcepts`? It's because opposite of "forward-looking" can be many things, eg, "short-sighted", "dwelling in the past", or simply "not-forward looking". 

`train_spectrum` allows you to define the opposite concept that you need and score documents on the spectrum between the two.

#### `train_concept`

Identify and score the presence of a specific concept in documents.
```julia
index = build_index(docs)
concept = train_concept(index, "sustainability")
scores = score(index, concept)

# show top 5 docs
index.docs[first(sortperm(scores, rev = true), 5)]
# 5-element Vector{String}:
# ["focus on smart growth, sustainability and affordability are just as important as business development and should not be sacrificed for economic growth.", "SUSTAINABILITY OF CITY", "we need to plan for global climate change, water and energy programs must be robust", "Public transport and planned, sustainable, affordable growth are the most important issues.", "Make more conservation and sustainability efforts."]
```

#### `train_spectrum`

Evaluate documents on a spectrum defined by two contrasting concepts.

```julia
index = "..." # re-use the index from the previous example
# aigenerate_kwargs are passed directly to PromptingTools.aigenerate (see PromptingTools.jl docs)
spectrum = train_spectrum(index, ("forward-looking", "dwelling in the past"); 
  aigenerate_kwargs = (;model="gpt3t"))
scores = score(index, spectrum)

# show top 5 docs for "forward-looking" (spectrum 1, ie, the "highest" score)
index.docs[first(sortperm(scores, rev = true), 5)]
# 5-element Vector{String}:
# ["He is doing a great job. Setting planning for growth, transportation and mobility together is an excellent approach.", "PLAN FOR ACCELERATED GROWTH. CLIMATE CHANGES PROMISES TO DELIVER MORE COASTAL CRISIS AND POPULATION DISPLACEMENT. AUSTIN WILL EXPAND AS A RESULT. THINK BIG. THANK YOU FOR PRIORITIZING SMART GROWTH AND A DENSE URBAN LANDSCAPE.", "Austin will grow! Prioritize development and roadways.", "Affordable housing. Better planning for future. Mass transit (rail system that covers City wide.", "PLAN FOR THE FUTURE AND SUSTAINABLE GROWTH. STOP FOCUSING ON EXCLUSIVE SERVICES LIKE TOLL ROAD EXPANSION AND INSTEAD, PUSH FOR PROGRAMS WITH THE LARGEST BENEFIT FOR THE MOST PEOPLE IN THE FUTURE, LIKE A SUBWAY SYSTEM AND CITY-SPONSORED DISTRIBUTED SOLAR AND ELECTRIC VEHICLE NETWORK."]
```

> [!TIP]
> Choose `train_concept` for alignment/closeness to a single theme, and `train_spectrum` for the position on the arbitrary spectrum defined by the two polarizing themes. Each function enhances text analysis with its unique approach to understanding content of each text/snippet/document.

> [!TIP]
> Remember to `serialize` your trained concepts and spectra to the disk for future use. This will save you time and money when you need to restart the REPL session.

### Classify Documents

Maybe you have done some fuzzy discovery at different levels of `k` (4,10,20,50). Now you have an idea of what you want, but neither of these auto-generated topic levels provide that - what should you do? 

`train_classifier` is the answer. It allows you to create a set of specific labels, one of which will be assigned to each document. 

Difference against standard standard multi-class classification:
- It operates on "embeddings" of the documents, capturing the semantic essence of the text, while being much more cost-efficient than running `aiclassify` on each document.
- It can be unsupervised, so you don't need to provide any labels for the documents (but you can if you want to). If there are no documents provided, the LLM engine will generate its own training data.


```julia
index = "..." # re-use the index from the previous example
# Let's create a few labels inspired by the automatic topic detection
labels = ["Improving traffic situation", "Taxes and public funding",
    "Safety and community", "Other"]
  
# Adding label descriptions will improve the quality of generated documents:
labels_description = [
    "Survey responses around infrastructure, improving traffic situation and related",
    "Decreasing taxes and giving more money to the community",
    "Survey responses around Homelessness, general safety and community related topics",
    "Any other topics like environment, education, governance, etc."]

# Train the classifier - it will generate 20 document examples (5 for each label x 4 labels). Ideally, you should aim for 10-20 examples per label.
cls = train_classifier(index, labels; labels_description)

# Get scores for all documents
scores = score(index, cls)

# Best label for each document
best_labels = score(index, cls; return_labels = true)
```

If you want to plot these, you can create a new `topic_level` in your index and then plot it as usual.

```julia
# Notice we provide our classifier `cls` as the second argument
build_clusters!(index, cls; topic_level = "MyClusters")

# Let's plot our clusters:
plot(index; topic_level = "MyClusters", title = "My Custom Clusters")
```

> [!TIP]
>  Be careful that if all the scores are similar, it means that the classifier is not very confident about the classification. Ie, look out for scores around `1/number_of_labels`.

> [!TIP]
> When you provide a vector of labels, try to add some "catch all" category like "Other" or "Not sure" to catch the documents that don't fit any of the provided labels.

### Create Custom Topic Levels

Both `build_clusters!` and `plot` expose a keyword argument to create a new named `topic_level` in the `index`.

```julia
build_clusters!(index, cls; topic_level = "MyClusters")
# Note: If not `topic_level` is provided, it will default to "Custom_1".

# Check what topic_levels are available
topic_levels(index) |> keys
```

You can create your custom topic levels without a classifier by directly providing the document `assignments` (ie, the cluster number for each document).

```julia
build_clusters!(index, assignments; topic_level = "MyClusters2", labels=["Label 1", "Label 2", "Label 3",...])
```

Then you can simply plot them via
```julia
plot(index; topic_level = "MyClusters2", title = "My Custom Clusters #2")
```

### Advanced Features and Best Practices
This section covers more advanced use cases and best practices for optimal results.

- Serialize your index to the disk (once the topics are fitted)! Saves money and time when you need to restart the REPL session.
- If you dislike the INFO logs, set `verbose=false`.
- Start by "zooming out" to get a sense of the overall themes (set `k=4`), then "zoom in" to explore the sub-themes (eg, set `k=20`)
- Leverage the plot interactivity (`PlotlyJS` backend will display the actual texts and topic labels on hover).
- For diverse datasets like survey questions (eg, DataFrame with many columns), create a separate index for each open-ended question for easy comparison / switching back and forth.
- For large documents, use `split_by_length` from PromptingTools.jl to split them into smaller chunks and explore the sub-themes.
- The package is designed for tens of thousands of documents, typically processed within minutes. For hundreds of thousands of documents, please await future versions with enhanced efficiency features.

### Core Concepts
LLMTextAnalysis.jl revolves around the `DocIndex` struct, which stores document chunks, embeddings, and related data. Document embeddings are pivotal, capturing the semantic essence of text chunks. LLMs are then employed to categorize and label the emerging themes.

### FAQs and Troubleshooting
For answers to common questions and troubleshooting advice, please refer to the FAQ section in the docs or open an issue.
It also includes some directions for future development.

### Inspirations and Acknowledgements
The development of LLMTextAnalysis.jl drew inspiration from tools like [lilac](https://www.lilacml.com/), [Nomic Atlas](https://atlas.nomic.ai/), and the work of Linus Lee (see the [presentation at AI Engineer Summit 2023](https://www.youtube.com/watch?v=YvobVu1l7GI)).

### Similar Packages
- [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) is a comprehensive package for text processing and analysis, offering functionalities like tokenization, stemming, sentiment analysis, and topic modeling. Unlike LLMTextAnalysis.jl, TextAnalysis.jl provides a broader range of traditional NLP tools suitable for various text analysis tasks.
- [TextModels.jl](https://github.com/JuliaText/TextModels.jl) enhances the TextAnalysis package with practical natural language models, typically based on neural networks (in Flux.jl)
- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) provides access to the HuggingFace Transformers library, which offers a wide range of pre-trained state-of-the-art models for NLP tasks. It also allows users to build transformer-based models from scratch on top Flux.jl.
- [StringAnalysis.jl](https://github.com/zgornel/StringAnalysis.jl) is a fork of TextAnalytics.jl, which offers a similar set of functionalities as TextAnalysis.jl, but with a slightly different API. It extends the original package with additional features like dimensionality reduction, semantic analysis, and more.
