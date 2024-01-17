# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

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