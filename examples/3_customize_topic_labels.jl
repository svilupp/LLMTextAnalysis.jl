# # Example 3: Customize Topic Labels
# The focus of this tutorial is how to customize the topic labels in the plot.
# 
# Necessary imports
using Downloads, CSV, DataFrames
using Plots
using LLMTextAnalysis
using PromptingTools
const PT = PromptingTools # for the templating functionality
plotlyjs(); # recommended backend for interactivity, install with `using Pkg; Pkg.add("PlotlyJS")`

# ## Customizing Topic Labels

# The topic labels are generated via a "template" (keyword argument `label_template` for the labeling function). 
#
# We default to "TopicLabelerBasic" (saved in `templates/topic-metadata/TopicLabelerBasic.jl`), but you can change the template and pass it to `build_clusters!()` or `plot()` via the `labeler_kwargs` keyword argument (eg, `labeler_kwargs = (; label_template=:MyTopicLabels)`).
#
# Why use the templates? We need to guarantee that you provide placeholders for `central_text`, `samples` and `keywords` in your template for everything to work, so we cannot simply accept any user-provided string. 
#
# Depending on how radically different your labels need to be, you have two choices: 
#
# 1) for smaller changes, use `instructions` placeholder in the `TopicLabelerWithInstructions` template. You should always start with this one.
# 2) for larger changes, you can create a new template, save it, load it to your template store and pass it to `labeler_kwargs` as mentioned above.

# ## Prepare the data and index

# Our setup - let's say we need our labels to be in French because our documents are.
#
# Note: I don't speak French, so apologies for any mistakes or NSFW content below
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

# Index as usual
index = build_index(french_sentences)

# ## Example of using `TopicLabelerWithInstructions`

# We need to change the label template to `TopicLabelerWithInstructions` and provide the instructions in the `labeler_kwargs` argument.

build_clusters!(
    index; k = 3,
    labeler_kwargs = (; label_template = :TopicLabelerWithInstructions,
        instructions = "All topic names must be translated to French.", model = "gpt3t"))

# Let's check the labels (3 refers to the number of clusters)
k = 3
index.topic_levels[k]
## 3-element Vector{TopicMetadata}:
##  TopicMetadata(ID: 1/3, Documents: 1, Label: "Étude du Comportement du Renard et du Chien",  Summary: -)
##  TopicMetadata(ID: 2/3, Documents: 3, Label: "Économie Financière Comportementale",  Summary: -)
##  TopicMetadata(ID: 3/3, Documents: 16, Label: "Les Proverbes de la Vie",  Summary: -)

# Note: I don't speak French, so I'm not sure if the labels are correct. It might take some tweaking of the instructions to get the labels right.

# If needed, you can always tweak the labels manually
index.topic_levels[k][1].label = "Étude du Comportement du Renard et du Chien - 123456789"

# How would we find available templates? `aitemplates("Labeler")`!

# We would see the templates with partial match, what they do and the PLACEHOLDERS they require, eg, 
#
## PromptingTools.AITemplateMetadata
##   name: Symbol TopicLabelerWithInstructions
##   description: String "Advanced labeler for a given topic/theme in 2-5 words based on the provided central text, samples and keywords. It provides a field for special instructions. If you do not have any special instructions, provide `instructions=\"None.\"`. Placeholders: `central_text`, `samples`, `keywords`, `instructions`"
##   version: String "1.0"
##   wordcount: Int64 653
##   variables: Array{Symbol}((4,))
##   system_preview: String "Act as a world-class behavioural researcher, unbiased and trained to surface key underlying themes.\n"
##   user_preview: String "###Central Text###\n{{central_text}}\n\n###Sample Texts###\n{{samples}}\n\n###Common Words###\n{{keywords}}"
##   source: String ""

# ## Example of creating a new template - the easy way
# There is a simpler way to create a new template. You can use the `PT.create_template(; user="..", system="..", load_as="..")` to create and load a template in a single function call.
#
# Example:
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

# You could inspect that tpl is a vector of UserMessage and SystemMessage, but it's not necessary. We can use it directly in `build_clusters!` or `plot` as `label_template = :MyTopicLabels`.
build_clusters!(
    index; k = 3, labeler_kwargs = (; label_template = :MyTopicLabels, model = "gpt3t"))
pl = plot(index; k = 3,
    title = "Sujets d'actualité du 2024-02-13")

# If you want to understand what `create_template` does under-the-hood, follow this step by step walkthrough below.
#
# Note: Templates created with `create_template` are NOT saved in the `templates` and they will disappear after your restart REPL.
# If you want to save it in your project permanently, use `PT.save_template` as shown in the next example.

# ## Example of creating a new template - the long way

# We first duplicate the `TopicLabelerBasic` template to have a starting point and 2 add two new instructions in section "Topic Name Instructions".

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

    The most suitable topic name is:""")]
filename = joinpath(pkgdir(LLMTextAnalysis), "templates",
    "topic-metadata",
    "MyTopicLabels.json")
PT.save_template(filename,
    tpl_custom;
    version = "1.0",
    description = "My custom template for topic labeling. Placeholders: `central_text`, `samples`, `keywords`")

# Note: The filename of the template is "MyTopicLabels.json", so the template will be accessible in the code as `:MyTopicLabels` (symbols signify the use of templates).
#
# We need to explicitly re-load all templates with `load_templates!()` to make the new template available for use.

load_templates!()

# Then to force overwrite the existing labels with the new template, we call `build_clusters!` with the `labeler_kwargs` argument.
#
# Notice that we provide the label_template argument that matches the file name of the template we created
build_clusters!(
    index; k = 3, labeler_kwargs = (; label_template = :MyTopicLabels, model = "gpt3t"))
pl = plot(index; k = 3,
    title = "Sujets d'actualité du 2024-02-13")

# ## Difference between labeler_kwargs for `build_clusters!` and `plot`

# So when should I use `build_clusters!` and when should I simply use `plot`?
#
# It depends if you want to overwrite any existing topic labels with the provided `labeler_kwargs` or not:
#
# - `plot` call will only create new labels if you specify `k` (number of clusters) that **is NOT available yet** (ie, no key available in `index.topic_levels`).
# - `build_clusters!` will always overwrite the existing labels if you specify `k` (number of clusters).