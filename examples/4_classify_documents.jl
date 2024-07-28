# # Example 4: Classify Documents
# Simetimes you need to assign some specific labels to each document. You could use `aiclassify` and process each document separately, but that would be costly.
# `train_classifier` is a good alternative leveraging the embeddings in your document `index`.

# Necessary imports
using Downloads, CSV, DataFrames
using Plots
using LLMTextAnalysis
plotlyjs(); # plotlyjs() is the recommended backend for Plots.jl for interactivity, install with `using Pkg; Pkg.add("PlotlyJS")`

# ## Prepare the Data
# For this tutorial, we will use the [City of Austin's Community Survey](https://data.austintexas.gov/Health-and-Community-Services/2019-City-of-Austin-Community-Survey/s2py-ceb7).
#
# We will pick one open-ended question.
# Download the survey data
Downloads.download(
    "https://data.austintexas.gov/api/views/s2py-ceb7/rows.csv?accessType=DOWNLOAD",
    joinpath(@__DIR__, "cityofaustin.csv"));

# Read the survey data into a DataFrame
df = CSV.read(joinpath(@__DIR__, "cityofaustin.csv"), DataFrame);

# Let's select one of the open-ended questions, eg,
col = "Q25 - If there was one thing you could share with the Mayor regarding the City of Austin (any comment, suggestion, etc.), what would it be?"
docs = df[!, col] |> skipmissing |> collect;

# ## Build the Index
# Index the documents (ie, embed them)
index = build_index(docs)

# ## Classification
# Sometimes you need to assign some specific labels to each document.
#
# For these situations, `LLMTextAnalysis` offers `train_classifier` to train a classifier that will assign provided `labels` to new documents based on their content.
#
# The labels can be anything you want, but they should be descriptive of the content of the documents (there is a field `labels_description` to be able to provide more verbose descriptions).
#
# The resulting return type is a `TrainedClassifier` and when you `score` a document, you will get a score for each label (ie, a vector).
# If you score multiple documents, you'll get a matrix of scores, where each row is a document and each column is a label -> the best label is the one with the highest score in each row.
#
# Tip 1: Be careful that if all the scores are similar, it means that the classifier is not very confident about the classification. Ie, look out for scores around `1/number_of_labels`.
#
# Tip 2: When you provide a vector of labels, try to add some "catch all" category like "Other" or "Not sure" to catch the documents that don't fit any of the provided labels.
# 

# ### Classification Based on Labeled Examples
# Let's create a few labels inspired by the automatic topic detection
labels = ["Improving traffic situation", "Taxes and public funding",
    "Safety and community", "Other"]

# Let's say we have labeled a few documents - ideally, you should have 5-10 examples for EACH label
docs_ids = [1, 2674, 4, 17, 23, 69, 2669, 6]
docs_labels = [1, 1, 2, 2, 3, 3, 4, 4]

# Train the classifier
cls = train_classifier(index, labels; docs_ids, docs_labels)

# Get scores for all documents
scores = score(index, cls)
scores[1:3, :]
# Note: Watch out for scores around `1/number_of_labels` - it means the classifier is not very confident about the classification

# Best label for each document
label_ids = argmax(scores, dims = 2) |> vec |> x -> map(i -> i[2], x)
best_labels = cls.labels[label_ids]
best_labels[1:3]

# Or do it in one line with `return_labels=true`
best_labels = score(index, cls; return_labels = true)
best_labels[1:3]

# ### Classification Without Labeled Examples
# When we don't have any examples, we can ask an AI model to generate some potential examples for us.
# It might be less precise, but it can save us a lot of time.
#
# Adding label descriptions will improve the quality of generated documents:
labels_description = [
    "Survey responses around infrastructure, improving traffic situation and related",
    "Decreasing taxes and giving more money to the community",
    "Survey responses around Homelessness, general safety and community related topics",
    "Any other topics like environment, education, governance, etc."]

# Train the classifier - it will generate 20 document examples (5 for each label x 4 labels)
cls = train_classifier(index, labels; labels_description)

# Get scores for all documents
scores = score(index, cls)
scores[1:3, :]

# Best label for each document
best_labels = score(index, cls; return_labels = true)
best_labels[1:3]

# ## Adding Custom Topic Level to the Index

# Let's say we want to add a custom topic level to the index.
# We can do it by providing the trained classifier `cls` to the function `build_clusters!`.

build_clusters!(index, cls; topic_level = "MyClusters")
# Note: If not `topic_level` is provided, it will default to "Custom_1".

# Check what topic_levels are available
topic_levels(index) |> keys

# ## Plotting

# Whether you have auto-generated topics or custom topics, you can plot them with `plot` by leveraging the keyword argument `topic_level`.

# Let's plot our clusters:
plot(index; topic_level = "MyClusters", title = "My Custom Clusters")
