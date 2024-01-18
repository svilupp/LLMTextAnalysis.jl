using LLMTextAnalysis
using Documenter

DocMeta.setdocmeta!(LLMTextAnalysis,
    :DocTestSetup,
    :(using LLMTextAnalysis);
    recursive = true)

makedocs(;
    modules = [LLMTextAnalysis],
    authors = "J S <49557684+svilupp@users.noreply.github.com> and contributors",
    repo = "https://github.com/svilupp/LLMTextAnalysis.jl/blob/{commit}{path}#{line}",
    sitename = "LLMTextAnalysis.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/svilupp/LLMTextAnalysis.jl",
        canonical = "https://svilupp.github.io/LLMTextAnalysis.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = nothing),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Explore Topics in Your Documents" => "examples/1_topics_in_city_of_austin_community_survey.md",
            "Look for specific Concept/Spectrum" => "examples/2_concept_labeling_in_city_of_austin_community_survey.md",
        ],
        "F.A.Q." => "FAQ.md",
        "References" => "api_reference.md",
    ])

deploydocs(;
    repo = "github.com/svilupp/LLMTextAnalysis.jl",
    devbranch = "main")
