module LLMTextAnalysis

using Languages, Snowball, WordTokenizers
using UMAP, Clustering, Distances
using SparseArrays: sparse
using LinearAlgebra: normalize
using PromptingTools
using Random: shuffle
using Statistics: mean
using MLJLinearModels, Tables
const PT = PromptingTools

export load_templates!
# export nunique, sigmoid
include("utils.jl")

export DocIndex, TopicMetadata, TrainedConcept, TrainedSpectrum
include("types.jl")

export build_index, prepare_plot!
# export build_keywords
include("preparation.jl")

export build_clusters!
# export build_topic
include("topic_modelling.jl")

export cross_validate_accuracy, train_spectrum, train_concept, score
# export create_folds
include("concept_labeling.jl")

function __init__()
    ## Load extra templates
    load_templates!()
end

# TODO: Enable precompilation to reduce start time, disabled logging
# with_logger(NullLogger()) do
#     @compile_workload include("precompilation.jl")
# end
end
