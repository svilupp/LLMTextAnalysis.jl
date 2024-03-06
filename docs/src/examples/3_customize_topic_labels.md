```@meta
EditURL = "../../../examples/3_customize_topic_labels.jl"
```

# Example 3: Customize Topic Labels
The focus of this tutorial is how to customize the topic labels in the plot.

Necessary imports

````julia
using Downloads, CSV, DataFrames
import PlotlyJS, PlotlyDocumenter  ## Only for the documentation, not needed for users!
using LLMTextAnalysis
using PromptingTools
const PT = PromptingTools # for the templating functionality
````

````
PromptingTools
````

## Customizing Topic Labels

The topic labels are generated via a "template" (keyword argument `label_template` for the labeling function).

We default to "TopicLabelerBasic" (saved in `templates/topic-metadata/TopicLabelerBasic.jl`), but you can change the template and pass it to `build_clusters!()` or `PlotlyJS.plot()` via the `labeler_kwargs` keyword argument (eg, `labeler_kwargs = (; label_template=:MyTopicLabels)`).

Why use the templates? We need to guarantee that you provide placeholders for `central_text`, `samples` and `keywords` in your template for everything to work, so we cannot simply accept any user-provided string.

Depending on how radically different your labels need to be, you have two choices:

1) for smaller changes, use `instructions` placeholder in the `TopicLabelerWithInstructions` template. You should always start with this one.
2) for larger changes, you can create a new template, save it, load it to your template store and pass it to `labeler_kwargs` as mentioned above.

## Prepare the data and index

Our setup - let's say we need our labels to be in French because our documents are.

Note: I don't speak French, so apologies for any mistakes or NSFW content below

````julia
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
````

````
20-element Vector{String}:
 "Bonjour, comment ça va?"
 "Je m'appelle Marie."
 "Il fait beau aujourd'hui."
 "Quel est ton plat préféré?"
 "J'adore voyager dans le monde."
 "As-tu vu ce film?"
 "A quelle heure est le rendez-vous?"
 "Les chats sont mignons."
 "Tu veux sortir ce soir?"
 "C'est délicieux!"
 "Où se trouve la bibliothèque?"
 "Tu parles plusieurs langues?"
 "J'aime écouter de la musique."
 "Quel est ton sport préféré?"
 "Il est tard, tu devrais dormir."
 "Quel temps fait-il demain?"
 "Je suis très fatigué."
 "C'est une bonne idée."
 "Tu me manques."
 "Il faut rester positif."
````

Index as usual

````julia
index = build_index(french_sentences)
````

````
DocIndex(Documents: 20, PlotData: None, Topic Levels: None)
````

## Example of using `TopicLabelerWithInstructions`

We need to change the label template to `TopicLabelerWithInstructions` and provide the instructions in the `labeler_kwargs` argument.

````julia
build_clusters!(
    index; k = 3,
    labeler_kwargs = (; label_template = :TopicLabelerWithInstructions,
        instructions = "All topic names must be translated to French.", model = "gpt3t"))
````

````
DocIndex(Documents: 20, PlotData: None, Topic Levels: 3)
````

Let's check the labels (3 refers to the number of clusters)

````julia
k = 3
index.topic_levels[k]
# 3-element Vector{TopicMetadata}:
#  TopicMetadata(ID: 1/3, Documents: 1, Label: "Étude du Comportement du Renard et du Chien",  Summary: -)
#  TopicMetadata(ID: 2/3, Documents: 3, Label: "Économie Financière Comportementale",  Summary: -)
#  TopicMetadata(ID: 3/3, Documents: 16, Label: "Les Proverbes de la Vie",  Summary: -)
````

````
3-element Vector{TopicMetadata}:
 TopicMetadata(ID: 1/3, Documents: 13, Label: "Exploration des Conversations Françaises",  Summary: -)
 TopicMetadata(ID: 2/3, Documents: 5, Label: "Météo et Émotions",  Summary: -)
 TopicMetadata(ID: 3/3, Documents: 2, Label: ""Compréhension des Questions en Français"",  Summary: -)
````

Note: I don't speak French, so I'm not sure if the labels are correct. It might take some tweaking of the instructions to get the labels right.

If needed, you can always tweak the labels manually

````julia
index.topic_levels[k][1].label = "Étude du Comportement du Renard et du Chien - 123456789"
````

````
"Étude du Comportement du Renard et du Chien - 123456789"
````

How would we find available templates? `aitemplates("Labeler")`!

We would see the templates with partial match, what they do and the PLACEHOLDERS they require, eg,

````julia
# PromptingTools.AITemplateMetadata
#   name: Symbol TopicLabelerWithInstructions
#   description: String "Advanced labeler for a given topic/theme in 2-5 words based on the provided central text, samples and keywords. It provides a field for special instructions. If you do not have any special instructions, provide `instructions=\"None.\"`. Placeholders: `central_text`, `samples`, `keywords`, `instructions`"
#   version: String "1.0"
#   wordcount: Int64 653
#   variables: Array{Symbol}((4,))
#   system_preview: String "Act as a world-class behavioural researcher, unbiased and trained to surface key underlying themes.\n"
#   user_preview: String "###Central Text###\n{{central_text}}\n\n###Sample Texts###\n{{samples}}\n\n###Common Words###\n{{keywords}}"
#   source: String ""
````

## Example of creating a new template - the easy way
There is a simpler way to create a new template. You can use the `PT.create_template(; user="..", system="..", load_as="..")` to create and load a template in a single function call.

Example:

````julia
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
````

````
┌ Warning: Template MyTopicLabels already exists, overwriting!
└ @ PromptingTools ~/.julia/packages/PromptingTools/KF3MT/src/templates.jl:471

````

You could inspect that tpl is a vector of UserMessage and SystemMessage, but it's not necessary. We can use it directly in `build_clusters!` or `plot` as `label_template = :MyTopicLabels`.

````julia
build_clusters!(
    index; k = 3, labeler_kwargs = (; label_template = :MyTopicLabels, model = "gpt3t"))
pl = PlotlyJS.plot(index; k = 3,
    title = "Sujets d'actualité du 2024-02-13")
````

```@raw html
<div
    class="webio-mountpoint"
    data-webio-mountpoint="5878740521163038321"
>
    <script>
    (function(){
    // Some integrations (namely, IJulia/Jupyter) use an alternate render pathway than
    // just putting the html on the page. If WebIO isn't defined, then it's pretty likely
    // that we're in one of those situations and the integration just isn't installed
    // correctly.
    if (typeof window.WebIO === "undefined") {
        document
            .querySelector('[data-webio-mountpoint="5878740521163038321"]')
            .innerHTML = (
                '<div style="padding: 1em; background-color: #f8d6da; border: 1px solid #f5c6cb; font-weight: bold;">' +
                '<p><strong>WebIO not detected.</strong></p>' +
                '<p>Please read ' +
                '<a href="https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/" target="_blank">the troubleshooting guide</a> ' +
                'for more information on how to resolve this issue.</p>' +
                '<p><a href="https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/" target="_blank">https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/</a></p>' +
                '</div>'
            );
        return;
    }
    WebIO.mount(
        document.querySelector('[data-webio-mountpoint="5878740521163038321"]'),
        {"props":{},"nodeType":"Scope","type":"node","instanceArgs":{"imports":{"data":[{"name":"Plotly","type":"js","url":"\/assetserver\/46b0af2f03ca323adf531facbdab5be59e805cc6-plotly.min.js"},{"name":null,"type":"js","url":"\/assetserver\/103980130ce5b799e931b7ec1b8ad89411b10e20-plotly_webio.bundle.js"}],"type":"async_block"},"id":"10289561463356047098","handlers":{"_toImage":["(function (options){return this.Plotly.toImage(this.plotElem,options).then((function (data){return WebIO.setval({\"name\":\"image\",\"scope\":\"10289561463356047098\",\"id\":\"75\",\"type\":\"observable\"},data)}))})"],"__get_gd_contents":["(function (prop){prop==\"data\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"10289561463356047098\",\"id\":\"76\",\"type\":\"observable\"},this.plotElem.data)) : undefined; return prop==\"layout\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"10289561463356047098\",\"id\":\"76\",\"type\":\"observable\"},this.plotElem.layout)) : undefined})"],"_downloadImage":["(function (options){return this.Plotly.downloadImage(this.plotElem,options)})"],"_commands":["(function (args){var fn=args.shift(); var elem=this.plotElem; var Plotly=this.Plotly; args.unshift(elem); return Plotly[fn].apply(this,args)})"]},"systemjs_options":null,"mount_callbacks":["function () {\n    var handler = ((function (Plotly,PlotlyWebIO){PlotlyWebIO.init(WebIO); var gd=this.dom.querySelector(\"#plot-aff25175-da99-4143-af88-2b5eb7535ad4\"); this.plotElem=gd; this.Plotly=Plotly; (window.Blink!==undefined) ? (gd.style.width=\"100%\", gd.style.height=\"100vh\", gd.style.marginLeft=\"0%\", gd.style.marginTop=\"0vh\") : undefined; window.onresize=(function (){return Plotly.Plots.resize(gd)}); Plotly.newPlot(gd,[{\"mode\":\"markers\",\"y\":[4.9646997,5.109455,6.135841,5.8832126,4.6237655,4.988805,5.5108237,5.296445,6.168445,5.8728867,4.506696,5.116504,5.43381],\"type\":\"scatter\",\"name\":\"Quelles sont les expressions communes?\",\"customdata\":[\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\"],\"text\":[\"Bonjour, comment ça va?\",\"Je m'appelle Marie.\",\"Quel est ton plat préféré?\",\"J'adore voyager dans le monde.\",\"As-tu vu ce film?\",\"Les chats sont mignons.\",\"C'est délicieux!\",\"Tu parles plusieurs langues?\",\"J'aime écouter de la musique.\",\"Quel est ton sport préféré?\",\"Je suis très fatigué.\",\"C'est une bonne idée.\",\"Tu me manques.\"],\"hovertemplate\":\"<b>Topic:<\/b> Quelles sont les expressions communes?<br><b>Text:<\/b> %{text}<extra><\/extra>\",\"x\":[-5.258047,-6.279117,-5.4394016,-6.557244,-4.6042376,-6.994609,-7.0774918,-4.5895214,-6.1670322,-5.066315,-6.548118,-7.4561706,-5.9200873]},{\"mode\":\"markers\",\"y\":[4.144289,4.5024214,4.2770815,3.6553814,3.6649156],\"type\":\"scatter\",\"name\":\"Quel temps fait-il demain?\",\"customdata\":[\"\",\"\",\"\",\"\",\"\"],\"text\":[\"Il fait beau aujourd'hui.\",\"Tu veux sortir ce soir?\",\"Il est tard, tu devrais dormir.\",\"Quel temps fait-il demain?\",\"Il faut rester positif.\"],\"hovertemplate\":\"<b>Topic:<\/b> Quel temps fait-il demain?<br><b>Text:<\/b> %{text}<extra><\/extra>\",\"x\":[-6.1393723,-3.9519272,-5.5597258,-5.4869413,-6.629352]},{\"mode\":\"markers\",\"y\":[3.7639422,3.8556092],\"type\":\"scatter\",\"name\":\"Quelle heure est le rendez-vous ?\",\"customdata\":[\"\",\"\"],\"text\":[\"A quelle heure est le<br>rendez-vous?\",\"Où se trouve la bibliothèque?\"],\"hovertemplate\":\"<b>Topic:<\/b> Quelle heure est le rendez-vous ?<br><b>Text:<\/b> %{text}<extra><\/extra>\",\"x\":[-4.9080296,-4.3250093]}],{\"xaxis\":{\"title\":{\"text\":\"UMAP 1\"}},\"template\":\"plotly_white\",\"margin\":{\"l\":50,\"b\":50,\"r\":50,\"t\":60},\"title\":\"Sujets d'actualité du 2024-02-13\",\"yaxis\":{\"title\":{\"text\":\"UMAP 2\"}}},{\"showLink\":false,\"editable\":false,\"responsive\":true,\"staticPlot\":false,\"scrollZoom\":true}); gd.on(\"plotly_hover\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"hover\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"hover\",\"scope\":\"10289561463356047098\",\"id\":\"71\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_unhover\",(function (){return WebIO.setval({\"name\":\"hover\",\"scope\":\"10289561463356047098\",\"id\":\"71\",\"type\":\"observable\"},{})})); gd.on(\"plotly_selected\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"selected\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"selected\",\"scope\":\"10289561463356047098\",\"id\":\"72\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_deselect\",(function (){return WebIO.setval({\"name\":\"selected\",\"scope\":\"10289561463356047098\",\"id\":\"72\",\"type\":\"observable\"},{})})); gd.on(\"plotly_relayout\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"relayout\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"relayout\",\"scope\":\"10289561463356047098\",\"id\":\"74\",\"type\":\"observable\"},filtered_data.out)) : undefined})); return gd.on(\"plotly_click\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"click\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"click\",\"scope\":\"10289561463356047098\",\"id\":\"73\",\"type\":\"observable\"},filtered_data.out)) : undefined}))}));\n    (WebIO.importBlock({\"data\":[{\"name\":\"Plotly\",\"type\":\"js\",\"url\":\"\/assetserver\/46b0af2f03ca323adf531facbdab5be59e805cc6-plotly.min.js\"},{\"name\":null,\"type\":\"js\",\"url\":\"\/assetserver\/103980130ce5b799e931b7ec1b8ad89411b10e20-plotly_webio.bundle.js\"}],\"type\":\"async_block\"})).then((imports) => handler.apply(this, imports));\n}\n"],"observables":{"_toImage":{"sync":false,"id":"78","value":{}},"hover":{"sync":false,"id":"71","value":{}},"selected":{"sync":false,"id":"72","value":{}},"__gd_contents":{"sync":false,"id":"76","value":{}},"click":{"sync":false,"id":"73","value":{}},"image":{"sync":true,"id":"75","value":""},"__get_gd_contents":{"sync":false,"id":"80","value":""},"_downloadImage":{"sync":false,"id":"79","value":{}},"relayout":{"sync":false,"id":"74","value":{}},"_commands":{"sync":false,"id":"77","value":[]}}},"children":[{"props":{"id":"plot-aff25175-da99-4143-af88-2b5eb7535ad4"},"nodeType":"DOM","type":"node","instanceArgs":{"namespace":"html","tag":"div"},"children":[]}]},
        window,
    );
    })()
    </script>
</div>

```

If you want to understand what `create_template` does under-the-hood, follow this step by step walkthrough below.

Note: Templates created with `create_template` are NOT saved in the `templates` and they will disappear after your restart REPL.
If you want to save it in your project permanently, use `PT.save_template` as shown in the next example.

## Example of creating a new template - the long way

We first duplicate the `TopicLabelerBasic` template to have a starting point and 2 add two new instructions in section "Topic Name Instructions".

````julia
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
````

````
"/Users/simljx/Documents/LLMTextAnalysis/templates/topic-metadata/MyTopicLabels.json"
````

Note: The filename of the template is "MyTopicLabels.json", so the template will be accessible in the code as `:MyTopicLabels` (symbols signify the use of templates).

We need to explicitly re-load all templates with `load_templates!()` to make the new template available for use.

````julia
load_templates!()
````

Then to force overwrite the existing labels with the new template, we call `build_clusters!` with the `labeler_kwargs` argument.

Notice that we provide the label_template argument that matches the file name of the template we created

````julia
build_clusters!(
    index; k = 3, labeler_kwargs = (; label_template = :MyTopicLabels, model = "gpt3t"))
pl = PlotlyJS.plot(index; k = 3,
    title = "Sujets d'actualité du 2024-02-13")
````

```@raw html
<div
    class="webio-mountpoint"
    data-webio-mountpoint="16864504510432645187"
>
    <script>
    (function(){
    // Some integrations (namely, IJulia/Jupyter) use an alternate render pathway than
    // just putting the html on the page. If WebIO isn't defined, then it's pretty likely
    // that we're in one of those situations and the integration just isn't installed
    // correctly.
    if (typeof window.WebIO === "undefined") {
        document
            .querySelector('[data-webio-mountpoint="16864504510432645187"]')
            .innerHTML = (
                '<div style="padding: 1em; background-color: #f8d6da; border: 1px solid #f5c6cb; font-weight: bold;">' +
                '<p><strong>WebIO not detected.</strong></p>' +
                '<p>Please read ' +
                '<a href="https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/" target="_blank">the troubleshooting guide</a> ' +
                'for more information on how to resolve this issue.</p>' +
                '<p><a href="https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/" target="_blank">https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/</a></p>' +
                '</div>'
            );
        return;
    }
    WebIO.mount(
        document.querySelector('[data-webio-mountpoint="16864504510432645187"]'),
        {"props":{},"nodeType":"Scope","type":"node","instanceArgs":{"imports":{"data":[{"name":"Plotly","type":"js","url":"\/assetserver\/46b0af2f03ca323adf531facbdab5be59e805cc6-plotly.min.js"},{"name":null,"type":"js","url":"\/assetserver\/103980130ce5b799e931b7ec1b8ad89411b10e20-plotly_webio.bundle.js"}],"type":"async_block"},"id":"15895439922540052648","handlers":{"_toImage":["(function (options){return this.Plotly.toImage(this.plotElem,options).then((function (data){return WebIO.setval({\"name\":\"image\",\"scope\":\"15895439922540052648\",\"id\":\"85\",\"type\":\"observable\"},data)}))})"],"__get_gd_contents":["(function (prop){prop==\"data\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"15895439922540052648\",\"id\":\"86\",\"type\":\"observable\"},this.plotElem.data)) : undefined; return prop==\"layout\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"15895439922540052648\",\"id\":\"86\",\"type\":\"observable\"},this.plotElem.layout)) : undefined})"],"_downloadImage":["(function (options){return this.Plotly.downloadImage(this.plotElem,options)})"],"_commands":["(function (args){var fn=args.shift(); var elem=this.plotElem; var Plotly=this.Plotly; args.unshift(elem); return Plotly[fn].apply(this,args)})"]},"systemjs_options":null,"mount_callbacks":["function () {\n    var handler = ((function (Plotly,PlotlyWebIO){PlotlyWebIO.init(WebIO); var gd=this.dom.querySelector(\"#plot-05b89368-459b-43fa-b017-9fb5b7e151c6\"); this.plotElem=gd; this.Plotly=Plotly; (window.Blink!==undefined) ? (gd.style.width=\"100%\", gd.style.height=\"100vh\", gd.style.marginLeft=\"0%\", gd.style.marginTop=\"0vh\") : undefined; window.onresize=(function (){return Plotly.Plots.resize(gd)}); Plotly.newPlot(gd,[{\"mode\":\"markers\",\"y\":[4.9646997,5.109455,6.135841,5.8832126,4.6237655,4.988805,5.5108237,5.296445,6.168445,5.8728867,4.506696,5.116504,5.43381],\"type\":\"scatter\",\"name\":\"Quelle est ta préférence alimentaire?\",\"customdata\":[\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\"],\"text\":[\"Bonjour, comment ça va?\",\"Je m'appelle Marie.\",\"Quel est ton plat préféré?\",\"J'adore voyager dans le monde.\",\"As-tu vu ce film?\",\"Les chats sont mignons.\",\"C'est délicieux!\",\"Tu parles plusieurs langues?\",\"J'aime écouter de la musique.\",\"Quel est ton sport préféré?\",\"Je suis très fatigué.\",\"C'est une bonne idée.\",\"Tu me manques.\"],\"hovertemplate\":\"<b>Topic:<\/b> Quelle est ta préférence alimentaire?<br><b>Text:<\/b> %{text}<extra><\/extra>\",\"x\":[-5.258047,-6.279117,-5.4394016,-6.557244,-4.6042376,-6.994609,-7.0774918,-4.5895214,-6.1670322,-5.066315,-6.548118,-7.4561706,-5.9200873]},{\"mode\":\"markers\",\"y\":[4.144289,4.5024214,4.2770815,3.6553814,3.6649156],\"type\":\"scatter\",\"name\":\"Quel temps fait-il aujourd'hui?\",\"customdata\":[\"\",\"\",\"\",\"\",\"\"],\"text\":[\"Il fait beau aujourd'hui.\",\"Tu veux sortir ce soir?\",\"Il est tard, tu devrais dormir.\",\"Quel temps fait-il demain?\",\"Il faut rester positif.\"],\"hovertemplate\":\"<b>Topic:<\/b> Quel temps fait-il aujourd'hui?<br><b>Text:<\/b> %{text}<extra><\/extra>\",\"x\":[-6.1393723,-3.9519272,-5.5597258,-5.4869413,-6.629352]},{\"mode\":\"markers\",\"y\":[3.7639422,3.8556092],\"type\":\"scatter\",\"name\":\"A Quelle Heure est le Rendez-vous?\",\"customdata\":[\"\",\"\"],\"text\":[\"A quelle heure est le<br>rendez-vous?\",\"Où se trouve la bibliothèque?\"],\"hovertemplate\":\"<b>Topic:<\/b> A Quelle Heure est le Rendez-vous?<br><b>Text:<\/b> %{text}<extra><\/extra>\",\"x\":[-4.9080296,-4.3250093]}],{\"xaxis\":{\"title\":{\"text\":\"UMAP 1\"}},\"template\":\"plotly_white\",\"margin\":{\"l\":50,\"b\":50,\"r\":50,\"t\":60},\"title\":\"Sujets d'actualité du 2024-02-13\",\"yaxis\":{\"title\":{\"text\":\"UMAP 2\"}}},{\"showLink\":false,\"editable\":false,\"responsive\":true,\"staticPlot\":false,\"scrollZoom\":true}); gd.on(\"plotly_hover\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"hover\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"hover\",\"scope\":\"15895439922540052648\",\"id\":\"81\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_unhover\",(function (){return WebIO.setval({\"name\":\"hover\",\"scope\":\"15895439922540052648\",\"id\":\"81\",\"type\":\"observable\"},{})})); gd.on(\"plotly_selected\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"selected\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"selected\",\"scope\":\"15895439922540052648\",\"id\":\"82\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_deselect\",(function (){return WebIO.setval({\"name\":\"selected\",\"scope\":\"15895439922540052648\",\"id\":\"82\",\"type\":\"observable\"},{})})); gd.on(\"plotly_relayout\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"relayout\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"relayout\",\"scope\":\"15895439922540052648\",\"id\":\"84\",\"type\":\"observable\"},filtered_data.out)) : undefined})); return gd.on(\"plotly_click\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"click\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"click\",\"scope\":\"15895439922540052648\",\"id\":\"83\",\"type\":\"observable\"},filtered_data.out)) : undefined}))}));\n    (WebIO.importBlock({\"data\":[{\"name\":\"Plotly\",\"type\":\"js\",\"url\":\"\/assetserver\/46b0af2f03ca323adf531facbdab5be59e805cc6-plotly.min.js\"},{\"name\":null,\"type\":\"js\",\"url\":\"\/assetserver\/103980130ce5b799e931b7ec1b8ad89411b10e20-plotly_webio.bundle.js\"}],\"type\":\"async_block\"})).then((imports) => handler.apply(this, imports));\n}\n"],"observables":{"_toImage":{"sync":false,"id":"88","value":{}},"hover":{"sync":false,"id":"81","value":{}},"selected":{"sync":false,"id":"82","value":{}},"__gd_contents":{"sync":false,"id":"86","value":{}},"click":{"sync":false,"id":"83","value":{}},"image":{"sync":true,"id":"85","value":""},"__get_gd_contents":{"sync":false,"id":"90","value":""},"_downloadImage":{"sync":false,"id":"89","value":{}},"relayout":{"sync":false,"id":"84","value":{}},"_commands":{"sync":false,"id":"87","value":[]}}},"children":[{"props":{"id":"plot-05b89368-459b-43fa-b017-9fb5b7e151c6"},"nodeType":"DOM","type":"node","instanceArgs":{"namespace":"html","tag":"div"},"children":[]}]},
        window,
    );
    })()
    </script>
</div>

```

## Difference between labeler_kwargs for `build_clusters!` and `plot`

So when should I use `build_clusters!` and when should I simply use `plot`?

It depends if you want to overwrite any existing topic labels with the provided `labeler_kwargs` or not:

- `plot` call will only create new labels if you specify `k` (number of clusters) that **is NOT available yet** (ie, no key available in `index.topic_levels`).
- `build_clusters!` will always overwrite the existing labels if you specify `k` (number of clusters).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

