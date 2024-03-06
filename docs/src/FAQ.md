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
    index = build_index(docs);
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

If you call `plotlyjs()` as well, the `Plots.plot()` will be interactive with the PlotlyJS backend. Using `plotlyjs()` requires having `PlotlyJS` added to your project (not imported, but added)!

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

## How to customize the topic labels?

The topic labels are generated via a "template" (keyword argument `label_template` for the labeling function). 

We default to "TopicLabelerBasic" (saved in `templates/topic-metadata/TopicLabelerBasic.jl`), but you can change the template and pass it to `build_clusters!()` or `plot()` via the `labeler_kwargs` keyword argument (eg, `labeler_kwargs = (; label_template=:MyTopicLabels)`).

Why use the templates? We need to guarantee that you provide placeholders for `central_text`, `samples` and `keywords` in your template for everything to work, so we cannot simply accept any user-provided string. 

Depending on how radically different your labels need to be, you have two choices: 

1) for smaller changes, use `instructions` placeholder in the `TopicLabelerWithInstructions` template. You should always start with this one.
2) for larger changes, you can create a new template, save it, load it to your template store and pass it to `labeler_kwargs` as mentioned above.

Our setup - let's say we need our labels to be in French because our documents are.

```julia
using PromptingTools
const PT = PromptingTools

## Note: I don't speak French, so apologies for any mistakes or NSFW content below
french_sentences = [
    "Bonjour, comment ça va?",
    "Je m'appelle Marie.",
    "Il fait beau aujourd'hui.",
    "Quel est ton plat préféré?",
    "J'adore voyager dans le monde.",
    "As-tu vu ce film?",
    "A quelle heure est le rendez-vous?",
    "Les chats sont mignons.",
    "Tu veux sortir ce soir?",
    "C'est délicieux!",
    "Où se trouve la bibliothèque?",
    "Tu parles plusieurs langues?",
    "J'aime écouter de la musique.",
    "Quel est ton sport préféré?",
    "Il est tard, tu devrais dormir.",
    "Quel temps fait-il demain?",
    "Je suis très fatigué.",
    "C'est une bonne idée.",
    "Tu me manques.",
    "Il faut rester positif."
]

## Index as usual
index = build_index(french_sentences)
```

### Example of using `TopicLabelerWithInstructions`

We need to change the label template to `TopicLabelerWithInstructions` and provide the instructions in the `labeler_kwargs` argument.

```julia
build_clusters!(
    index; k = 3,
    labeler_kwargs = (; label_template = :TopicLabelerWithInstructions,
        instructions = "All topic names must be translated to French.", model = "gpt3t"))

## Let's check the labels (3 refers to the number of clusters)
k=3
index.topic_levels[k]
## 3-element Vector{TopicMetadata}:
##  TopicMetadata(ID: 1/3, Documents: 1, Label: "Étude du Comportement du Renard et du Chien",  Summary: -)
##  TopicMetadata(ID: 2/3, Documents: 3, Label: "Économie Financière Comportementale",  Summary: -)
##  TopicMetadata(ID: 3/3, Documents: 16, Label: "Les Proverbes de la Vie",  Summary: -)

# Note: I don't speak French, so I'm not sure if the labels are correct. It might take some tweaking of the instructions to get the labels right.

# If needed, you can always tweak the labels manually
index.topic_levels[k][1].label = "Étude du Comportement du Renard et du Chien - 123456789"
```

How would we find available templates? `aitemplates("Labeler")`!

We would see the templates with a partial match, what they do and the PLACEHOLDERS they require, eg, 
```plaintext
PromptingTools.AITemplateMetadata
  name: Symbol TopicLabelerWithInstructions
  description: String "Advanced labeler for a given topic/theme in 2-5 words based on the provided central text, samples and keywords. It provides a field for special instructions. If you do not have any special instructions, provide `instructions=\"None.\"`. Placeholders: `central_text`, `samples`, `keywords`, `instructions`"
  version: String "1.0"
  wordcount: Int64 653
  variables: Array{Symbol}((4,))
  system_preview: String "Act as a world-class behavioural researcher, unbiased and trained to surface key underlying themes.\n"
  user_preview: String "###Central Text###\n{{central_text}}\n\n###Sample Texts###\n{{samples}}\n\n###Common Words###\n{{keywords}}"
  source: String ""
```

### Example of creating a new template

There is a simpler way to create a new template and immediately load it. 
You can use the `PT.create_template(; user="..", system="..", load_as="..")` to create and load a template in a single function call.

Example:

```julia
tpl = PT.create_template(;
    system = """
Act as a world-class behavioural researcher, unbiased and trained to surface key underlying themes.

Your task is create a topic name based on the provided information and sample texts.

**Topic Name Instructions:**
- A short phrase, ideally 2-5 words.
- Descriptive of the information provided.
- Brief and concise.
- Title Cased.
- Must be in French.
- Must be a question.
""",
    user = """
    ###Central Text###
    {{central_text}}

    ###Sample Texts###
    {{samples}}

    ###Common Words###
    {{keywords}}

    The most suitable topic name is:""",
    load_as = "MyTopicLabels");
```

You could inspect that `tpl` is a vector of UserMessage and SystemMessage, but it's not necessary. We can use it directly in `build_clusters!` or `plot` as `label_template = :MyTopicLabels`.

```julia
build_clusters!(
    index; k = 3, labeler_kwargs = (; label_template = :MyTopicLabels, model = "gpt3t"))
pl = plot(index; k = 3,
    title = "Sujets d'actualité du 2024-02-13")
```

If you want to understand what `create_template` does under-the-hood, follow this step by step walkthrough below.

Note: Templates created with `create_template` are NOT saved in the `templates` and they will disappear after your restart REPL.
If you want to save it in your project permanently, use `PT.save_template` as shown in the next example.

Let's now create a new template from scratch without the `create_template` function.

We first duplicate the `TopicLabelerBasic` template to have a starting point and add two new instructions in the section "Topic Name Instructions".

```julia
tpl_custom = [
    PT.SystemMessage("""
Act as a world-class behavioural researcher, unbiased and trained to surface key underlying themes.

Your task is create a topic name based on the provided information and sample texts.

**Topic Name Instructions:**
- A short phrase, ideally 2-5 words.
- Descriptive of the information provided.
- Brief and concise.
- Title Cased.
- Must be in French.
- Must be a question.
"""),
    PT.UserMessage("""
    ###Central Text###
    {{central_text}}

    ###Sample Texts###
    {{samples}}

    ###Common Words###
    {{keywords}}

    The most suitable topic name is:",
    """)]
filename = joinpath("templates",
    "topic-metadata",
    "MyTopicLabels.json")
PT.save_template(filename,
    tpl_custom;
    version = "1.0",
    description = "My custom template for topic labeling. Placeholders: `central_text`, `samples`, `keywords`")
```
Note: The filename of the template is "MyTopicLabels.json", so the template will be accessible in the code as `:MyTopicLabels` (symbols signify the use of templates).

We need to explicitly re-load all templates with `load_templates!()` to make the new template available for use.

```julia
load_templates!()
```

Then to force overwrite the existing labels with the new template, we call `build_clusters!` with the `labeler_kwargs` argument.

```julia
# Notice that we provide the label_template argument that matches the file name of the template we created
build_clusters!(
    index; k = 3, labeler_kwargs = (; label_template = :MyTopicLabels, model = "gpt3t"))
pl = plot(index; k = 3,
    title = "Sujets d'actualité du 2024-02-13")
```

### Difference between `labeler_kwargs` for `build_clusters!` and `plot`

So when should I use `build_clusters!` and when should I simply use `plot`?

It depends if you want to overwrite any existing topic labels with the provided `labeler_kwargs` or not:

- `plot` call will only create new labels if you specify `k` (number of clusters) that **is NOT available yet** (ie, no key available in `index.topic_levels`).
- `build_clusters!` will always overwrite the existing labels if you specify `k` (number of clusters).

## What's next for this package?

There are a few different functionalities that we're working on:
- [ ] Deduplicate code between PromptingTools and this package
- [ ] Build topic tree hierarchy, ie, layout the hierarchical relationships between topics (ie, at different `k` values)
- [ ] Scale to millions of data points