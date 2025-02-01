# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [0.8.0]

### Added
- Increased compatibility with `PromptingTools` upto 0.71.


## [0.7.0]

### Added
- Increased compatibility with `PromptingTools` upto 0.64.

## [0.6.0]

### Added
- Added a new template `TopicLabelerQuestionsWithContext` to generate topic labels for questions in a given context (eg, chatbot inputs).
- Added a new function `topic_tree` to generate a topic tree for a given topic level (display with `print_tree`).

### Updated
- Updated to use `PromptingTools` 0.44.0.
- That implicitly changes the default chat model to `gpt-4o-mini`.
- Minor updates to labeling templates to ensure that the labels are plain text, no markdown or code.
- Re-formatted code to SciML style guide.
- Upstreamed function `wrap_string` from `LLMTextAnalysis` to `PromptingTools`.

### Fixed
- Fixed a bug where `wrap_string` would not properly handle large words with a lot of Unicode characters.

## [0.5.0]

### Added
- Added a classification function `train_classifier` to train a model to classify documents into a set of predefined labels (as opposed to the more open-ended topic modeling in `build_clusters!`). You can either provide a small set of labeled documents to train the model (that are in the `index`), or just specify the `num_samples` and the LLM model will generate its own training data based on the `labels` and `labels_description` provided.
- Added a new template `TextWriterFromLabel` to generate synthetic documents for any given label (=topic).
- Added methods for `build_clusters!` to add custom topic levels, eg, from a `TrainedClassifier` (`build_clusters!(index,cls; topic_level="MyTopics")`) or directly via providing a vector of document `assignments` (`build_clusters!(index, assignments; topic_level="MyTopics")`). The convention is to use `topic_level::Integer` for auto-generated topics, and `topic_level::String` for custom topics.

### Updated
- Updated to use `PromptingTools` 0.15.

### Fixed
- Fixed a bug where `keywords` were not properly filtered before being provided to the auto-labeling function.

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