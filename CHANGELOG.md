# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [0.4.0]

### Added
- Added a new example on topic label customization (`examples/3_customize_topic_labels.jl`) and the corresponding sections in the FAQ.
- Added a few string cleanup tricks in `build_topic` function to strip unnecessary repetition of the prompt template in the generated labels.
- Added new templates `TopicLabelerWithInstructions` and `TopicSummarizerWithInstructions` that include the placeholder `instructions` to allow users to easily customize the labels and summaries, respectively.

### Fixed
- Fixed small typos in templates `TopicLabelerBasic` and `TopicSummarizerBasic`.

### Updated
- Updated logic in the `plot` to ensure topic labels are generated only when necessary. Use `build_clusters!` to force the generation of topic labels, or `plot` to generate them only if necessary.
- Increased compatibility for PromptingTools to 0.12.

## [0.3.2]

### Fixed
- Fixed a bug where `plot()` would error with `UndefVarError(scores1)`.

## [0.3.1]

### Fixed
- `wrap_string` utility would error with SubString chunks. Now it works with any AbstractString type.

## [0.3.0]

### Added
- Changed compat for PromptingTools to 0.10.0 (with new default models! Ie, default embeddings will not match the previous version)

## [0.2.1]

### Fixed
- Updated documentation to show Example 2 for concept/spectrum training.

## [0.2.0]

### Added
- Added `train_concept`. Introduces the ability to train a model focusing on a single, specific concept within a collection of documents. This function helps in identifying and scoring the presence or intensity of the selected concept across the document set. Ideal for thematic studies, sentiment analysis, or tracking specific ideas in the text.
- Added `train_spectrum`. Adds functionality to analyze documents across a spectrum defined by two contrasting concepts. This feature allows for a comparative analysis, providing insights into how documents align or contrast with two polar themes or sentiments.
- Spectrum and concept can be plotted using `plot` function.
- Improved plotting support: 
  - Added package extension for `PlotlyJS` for `plot` function.
  - Enabled `plot` function to accept an arbitrary `hoverdata` table with information to be added to the tooltip for each document (expects Tables.jl-compatible data).

## [0.1.0]

### Added
- Initial release with support for Plots.jl and building and labeling topics