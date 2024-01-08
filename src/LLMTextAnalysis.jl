module LLMTextAnalysis

using Languages, Snowball, WordTokenizers
using UMAP, Clustering, Distances
using SparseArrays: sparse
using LinearAlgebra: normalize
using PromptingTools
using Statistics: mean
const PT = PromptingTools

# export nunique
include("utils.jl")

export DocIndex, TopicMetadata
include("types.jl")

export build_index, prepare_plot!
# export build_keywords
include("preparation.jl")

export build_clusters!
# export build_topic
include("topic_modelling.jl")

# TODO: finish concept labelling
# include("concept_labelling.jl")

function __init__()
    ## Load extra templates
    PT.load_templates!() # refresh base templates
    PT.load_templates!(joinpath(@__DIR__, "..", "templates"); remove_templates = false) # add our custom ones
end

# TODO: Enable precompilation to reduce start time, disabled logging
# with_logger(NullLogger()) do
#     @compile_workload include("precompilation.jl")
# end
end
