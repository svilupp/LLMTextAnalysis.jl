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


## What's next for this package?

There are a few different functionalities that we're working on:
- [ ] Label any arbitrary concept (eg, "politeness")
- [ ] Label texts across some spectrum (eg, "positive" to "negative", or "formal" to "informal")
- [ ] Build topic tree hierarchy, ie, layout the hierarchical relationships between topics (ie, at different `k` values)
- [ ] Scale to millions of data points