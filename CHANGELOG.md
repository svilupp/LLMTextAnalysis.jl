# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `train_concept`. Introduces the ability to train a model focusing on a single, specific concept within a collection of documents. This function helps in identifying and scoring the presence or intensity of the selected concept across the document set. Ideal for thematic studies, sentiment analysis, or tracking specific ideas in the text.
- Added `train_spectrum`. Adds functionality to analyze documents across a spectrum defined by two contrasting concepts. This feature allows for a comparative analysis, providing insights into how documents align or contrast with two polar themes or sentiments.

##### Usage Example for `train_spectrum`
```julia
spectrum_model = train_spectrum(index, ("optimistic", "pessimistic"))
```

---

These entries in the CHANGELOG concisely document the addition of the `train_concept` and `train_spectrum` functions, outlining their primary purposes and providing basic usage examples.

### Fixed


## [0.1.0]

### Added
- Initial release with support for Plots.jl and building and labeling topics