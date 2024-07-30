module LLMTextAnalysis

using Languages, Snowball, WordTokenizers
using UMAP, Clustering, Distances
using SparseArrays: sparse
using LinearAlgebra: normalize
using PromptingTools
using PromptingTools: wrap_string
using Random: shuffle
using Statistics: mean
using MLJLinearModels, Tables
using AbstractTrees
using AbstractTrees: print_tree
const PT = PromptingTools

using PromptingTools: load_templates!
export load_templates!
# export nunique, sigmoid, softmax
include("utils.jl")

export DocIndex, TopicMetadata, TrainedConcept, TrainedSpectrum, TrainedClassifier,
       topic_levels
include("types.jl")

export build_index, prepare_plot!
# export build_keywords
include("preparation.jl")

export cross_validate_accuracy, train_spectrum, train_concept, score
# export create_folds
include("concept_labeling.jl")

export train_classifier
include("classification.jl")

export build_clusters!
# export build_topic
include("topic_modelling.jl")

export print_tree, topic_tree
include("topic_trees.jl")

function __init__()
    ## Load extra templates
    PT.load_templates!(joinpath(@__DIR__, "..", "templates"); remember_path = true) # add our custom ones
end

# TODO: Enable precompilation to reduce start time, disabled logging
# with_logger(NullLogger()) do
#     @compile_workload include("precompilation.jl")
# end
end
