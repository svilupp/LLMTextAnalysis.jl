```@meta
CurrentModule = LLMTextAnalysis
```

# Frequently Asked Questions

## Do I need to use OpenAI models?

No! Download [Ollama.ai](https://ollama.ai) and follow the documentation for [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) to get started with open-source models that can run on your local machine.

## How big can my dataset be?

This release has been built with datasets around 10K documents, where the end-to-run takes 1-2 minutes. It's possible to run with larger datasets, but it will take a bit longer.

To support datasets with >1M documents, we'll need to make a few changes. 

Open an issue if you're interested in this feature!

## Hacks for bigger datasets

If you have a larger dataset, the biggest bottleneck will be the `UMAP` step calculating the positions of each point. 

If you don't want to wait, you can do a quick approximation with quick & dirty "PCA" (we leverage SVD in LinearAlgebra and truncate the "reconstruction" to achieve something similar). Ideally, you would pre-process your embeddings, but we skip that step here for simplicity.

```julia
using LLMTextAnalysis
using LinearAlgebra
using Serialization

# ... assumes you have the rest of the code as per tutorial 

# Leverage serialization to save time in the future
if !isfile("my_index.jls")
    index = build_index(docs;
        labeler_kwargs = (; model = "gpt3t"))
    build_clusters!(index; k = 20, labeler_kwargs = (; model = "gpt3t"))
    ## Skip UMAP as it's too slow, do a simple PCA-like approximation
    ## we should center the data first, often it is scaled as well but with normalized embeddings it should be okay
    centered_emb = index.embeddings' .- mean(index.embeddings', dims = 1)
    F = svd(centered_emb)
    index.plot_data = permutedims(F.U[:, 1:2] * Diagonal(F.S[1:2]))
    serialize("my_index.jls", index)
else
    index = deserialize("my_index.jls")
end
```

## Minimal example for interactive plotting

In general, we overload `Plots.plot()` and `PlotlyJS.plot()` for plotting. You need to import only one of them.

If you call `plotlyjs()` as well, the `Plots.plot()` will be interactive with PlotlyJS backend.

In the documentation, we need to use `using PlotlyJS, PlotlyDocumenter`, but that's only for the docsite (see `examples/1_topics_in_city_of_austin_community_survey.jl`).

Simple MWE: 

```julia
using Plots
using LLMTextAnalysis
plotlyjs();
docs = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "He who hesitates is lost.",
    "Beauty is in the eye of the beholder.",
    "Actions speak louder than words.",
    "Where there's a will, there's a way.",
    "A picture is worth a thousand words.",
    "Time flies when you're having fun.",
    "All is fair in love and war.",
    "A penny saved is a penny earned.",
    "Birds of a feather flock together.",
    "Don't count your chickens before they hatch.",
    "Easy come, easy go.",
    "Fortune favors the bold.",
    "Haste makes waste.",
    "Ignorance is bliss.",
    "It's never too late to learn.",
    "Knowledge is power.",
    "Laughter is the best medicine.",
    "Money doesn't grow on trees."
]
index = build_index(docs)
pl = plot(index;
    title = "My first plot",
    labeler_kwargs = (; model = "gpt3t",))
```

Explore adding more `hoverdata` to each point in the scatter, in my experience, it makes the plot more informative (see `?plot` for more details).

## What's next for this package?

There are a few different functionalities that we're working on:
- [ ] Deduplicate code between PromptingTools and this package
- [ ] Build topic tree hierarchy, ie, layout the hierarchical relationships between topics (ie, at different `k` values)
- [ ] Scale to millions of data points