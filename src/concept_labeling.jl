## This script defines the functions used to train the concept & spectrum labelling.
# - Required types: `TrainedConcept`, `TrainedSpectrum` (in srx/types.jl)
# - Main functions: `train_concept`, `train_spectrum`, `train!`
# - Supporting functions: `create_folds`, `cross_validate_accuracy`

"""
    create_folds(k::Int, data_size::Int)

Create `k` random folds from a dataset of size `data_size`.

# Arguments
- `k::Int`: Number of folds to create.
- `n_obs::Int`: Total number of observations in the dataset.

# Returns
- `Vector{SubArray}`: A vector of `k` vectors, each containing indices for a fold.

# Examples
```julia
# Create 4 folds from a dataset Xt
n_obs = size(Xt, 1)
folds = create_folds(4, n_obs)
```
"""
function create_folds(k::Integer, n_obs::Integer)
    @assert k>=1 "k must be greater than 0"
    @assert n_obs>=k "n_obs must be greater than k"
    indices = shuffle(1:n_obs)
    fold_size = cld(n_obs, k)
    return [@view(indices[i:min(i + fold_size - 1, n_obs)])
            for i in 1:fold_size:n_obs]
end

"""
    cross_validate_accuracy(X::AbstractMatrix{<:Number},
                            y::AbstractVector{<:Integer};
                            verbose::Bool = true,
                            k::Int = 4,
                            lambda::Real = 1e-5) -> Float64

Perform k-fold cross-validation on the dataset `(X, y)` using logistic regression 
and return the average classification accuracy.

# Arguments
- `X::AbstractMatrix`: The feature matrix (observarions x features).
- `y::AbstractVector{<:Integer}`: The target vector (+-1)
- `verbose::Bool` (optional): If `true`, prints the accuracy of each fold. Defaults to `true`.
- `k::Int` (optional): The number of folds for cross-validation. Defaults to 4.
- `lambda::Real` (optional): Regularization parameter for logistic regression. Defaults to 1e-5.

# Returns
- `Float64`: The average classification accuracy across all folds.

# Example
```julia
acc = cross_validate_accuracy(Xt, y; k = 4, lambda = 1e-2)
```
"""
function cross_validate_accuracy(X::AbstractMatrix,
        y::AbstractVector{<:Integer};
        verbose::Bool = true,
        k::Int = 5,
        lambda::Real = 1e-5)
    n_obs = size(X, 1)
    folds = create_folds(k, n_obs)
    accuracies = Vector{Float64}(undef, k)

    lk = ReentrantLock()
    Threads.@threads for i in 1:k
        test_indices = folds[i]
        train_indices = setdiff(1:n_obs, test_indices)

        @views X_train, y_train = X[train_indices, :], y[train_indices]
        @views X_test, y_test = X[test_indices, :], y[test_indices]

        model = LogisticRegression(lambda;
            fit_intercept = false,
            scale_penalty_with_samples = true)
        theta = fit(model, X_train, y_train)
        y_pred = vec(X_test * theta)

        correct = sum((y_pred[i] >= 0.0) == (y_test[i] .>= 0.0)
                      for i in eachindex(y_pred, y_test))
        lock(lk) do
            accuracies[i] = correct / length(y_test)
        end
    end
    verbose && @info "Accuracy across folds: $((accuracies))"
    return mean(accuracies)
end

"""
    train_concept(index::AbstractDocumentIndex,
                  concept::String;
                  num_samples::Int = 100, verbose::Bool = true,
                  rewriter_template::Symbol = :StatementRewriter,
                  lambda::Real = 1e-3, negatives_samples::Int = 1,
                  aigenerate_kwargs::NamedTuple = NamedTuple(),
                  aiembed_kwargs::NamedTuple = NamedTuple(),)

Train a model to identify and score a specific Concept (defined by the string `concept`) based on `num_samples documents from `index`.

We effectively identify the "direction" in the embedding space that represent this concept and develop a model to be able to score our documents against it.

This function focuses on a single Concept, as opposed to a Spectrum (see `train_spectrum`), to gauge its presence, strength, or manifestations in the documents.

See also: `train_spectrum`, `train!`, `score`

# Arguments
- `index::AbstractDocumentIndex`: An index containing the documents to be analyzed.
- `concept::String`: The concept to be analyzed within the documents.
- `num_samples::Int` (optional): The number of documents to sample from the index for training. Defaults to 100.
- `verbose::Bool` (optional): If `true`, prints detailed logs during the process. Defaults to `true`.
- `rewriter_template::Symbol` (optional): The template used for rewriting statements. Defaults to `:StatementRewriter`.
- `lambda::Real` (optional): Regularization parameter for logistic regression. Defaults to 1e-3
- `negatives_samples::Int` (optional): The number of negative examples to use for training per each positive sample. Defaults to 1.
- `aigenerate_kwargs::NamedTuple` (optional): Additional arguments for the `aigenerate` function. See `?aigenerate` for more details.
- `aiembed_kwargs::NamedTuple` (optional): Additional arguments for the `aiembed` function. See `?aiembed` for more details.

# Returns
- A `TrainedConcept` object containing the trained model, along with relevant information such as rewritten documents (`docs`), embeddings (`embeddings`), and model coefficients (`coeffs`).

# Example
```julia
# Assuming `index` is an existing document index
my_concept = "sustainability"
concept_model = train_concept(index, my_concept)
```

Show the top 5 highest scoring documents for the concept:
```julia
scores = score(index, concept)
index.docs[first(sortperm(scores, rev = true), 5)]
```

You can customize the training by passing additional arguments to the AI generation and embedding functions. For example, you can specify the model to use for generation and how many samples to use:

```julia
concept = train_concept(index,
    "action-oriented";
    num_samples = 50,
    aigenerate_kwargs = (; model = "gpt3t"))
```

This function leverages large language models to extract and analyze the presence and variations of a specific concept within a document corpus. It can be particularly useful in thematic studies, sentiment analysis, or trend identification in large collections of text.

For further analysis, you can inspect the rewritten documents and their embeddings:
```julia
# Concept-related rewritten documents
concept_model.docs

# Embeddings of the rewritten documents
concept_model.embeddings
```
"""
function train_concept(index::AbstractDocumentIndex,
        concept::String;
        num_samples::Int = 100, verbose::Bool = true,
        rewriter_template::Symbol = :StatementRewriter,
        lambda::Real = 1e-3, negatives_samples::Int = 1,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple(),)
    #
    source_doc_ids = shuffle(1:length(index.docs)) |> x -> first(x, num_samples)
    concept = TrainedConcept(; index_id = index.id,
        source_doc_ids,
        concept)
    train!(index, concept; verbose,
        rewriter_template,
        lambda, negatives_samples,
        aigenerate_kwargs,
        aiembed_kwargs)
end

"""
    train!(index::AbstractDocumentIndex,
           concept::TrainedConcept;
           verbose::Bool = true,
           overwrite::Bool = false,
           rewriter_template::Symbol = :StatementRewriter,
           lambda::Real = 1e-3, negatives_samples::Int = 1,
           aigenerate_kwargs::NamedTuple = NamedTuple(),
           aiembed_kwargs::NamedTuple = NamedTuple(),)

Refine or retrain a previously trained `TrainedConcept` model. 

This function can be used to update the concept model with new data, adjust parameters, or completely retrain it.

See also: `train_concept`, `score`

# Arguments
- `index::AbstractDocumentIndex`: The document index containing the documents for analysis.
- `concept::TrainedConcept`: The trained concept object to be refined or retrained.
- `verbose::Bool` (optional): If `true`, prints detailed logs during the process. Defaults to `true`.
- `overwrite::Bool` (optional): If `true`, existing training data in the concept will be overwritten. Defaults to `false`.
- `rewriter_template::Symbol` (optional): The template used for rewriting statements. Defaults to `:StatementRewriter`.
- `lambda::Real` (optional): Regularization parameter for logistic regression. Defaults to 1e-3.
- `negatives_samples::Int` (optional): The number of negative examples to use for training per each positive sample. Defaults to 1.
- `aigenerate_kwargs::NamedTuple` (optional): Additional arguments for the `aigenerate` function.
- `aiembed_kwargs::NamedTuple` (optional): Additional arguments for the `aiembed` function.

# Returns
- The updated `TrainedConcept` object with refined or new training.

# Example
```julia
# Assuming `index` and `concept` are pre-existing objects
concept = train!(index, concept, verbose = true, overwrite = true)
```

This function allows for continuous improvement and adaptation of a concept model to new data or analysis perspectives. It is particularly useful in dynamic environments where the underlying data or the concept of interest may evolve over time.
"""
function train!(index::AbstractDocumentIndex,
        concept::TrainedConcept;
        verbose::Bool = true,
        overwrite::Bool = false,
        rewriter_template::Symbol = :StatementRewriter,
        lambda::Real = 1e-3, negatives_samples::Int = 1,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple(),)
    @assert !isempty(concept.concept) "Concept must be non-empty! (Provided: $(concept.concept))"

    @assert index.id==concept.index_id "Index ID mismatch! (Provided Index: $(index.id), Expected: $(concept.index_id))"

    @assert !isempty(concept.source_doc_ids) "Source document IDs must be non-empty! (Provided: $(concept.source_doc_ids))"

    cost_tracker = Threads.Atomic{Float64}(0.0)

    if isnothing(concept.docs) || overwrite
        ## Rewrite a few statements
        verbose && @info "Rewriting $(length(concept.source_doc_ids)) documents..."
        cost_tracker = Threads.Atomic{Float64}(0.0)
        model = hasproperty(aigenerate_kwargs, :model) ? aigenerate_kwargs.model :
                PT.MODEL_CHAT
        concept.docs = asyncmap(concept.source_doc_ids) do i
            msg = aigenerate(rewriter_template; verbose = false,
                statement = index.docs[i],
                lens = concept.concept,
                aigenerate_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            replace(msg.content, "\"" => "") |> strip
        end
    end
    @assert length(concept.docs)==length(concept.source_doc_ids) "Number of documents mismatch! (Provided: $(length(concept.docs)), Expected: $(length(concept.source_doc_ids)))"

    if isnothing(concept.embeddings) || overwrite
        ## Embed them // assumes consistency, ie, documents have not changed
        verbose && @info "Embedding $(length(concept.source_doc_ids)) documents..."
        model = hasproperty(aiembed_kwargs, :model) ? aiembed_kwargs.model :
                PT.MODEL_EMBEDDING
        embeddings = let
            msg = aiembed(concept.docs, normalize; verbose = false, aiembed_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            msg.content
        end
        # We remove the embedding of the original source documents
        # To isolate the "direction" of the concept
        @info size(embeddings) size(index.embeddings)
        concept.embeddings = (embeddings .-
                              @view(index.embeddings[:, concept.source_doc_ids])) .|>
                             Float32
    end
    @assert size(concept.embeddings, 2)==length(concept.source_doc_ids) "Number of embeddings mismatch! (Provided: $(size(concept.embeddings, 2)), Expected: $(length(concept.source_doc_ids)))"

    verbose && cost_tracker[] > 0 &&
        @info "Done with LLMs. Total cost: \$$(round(cost_tracker[],digits=3))"

    ## Train a classifier // always retrain, it's super fast
    verbose && @info "Training a classifier..."
    model = LogisticRegression(lambda; fit_intercept = false)
    # use randomly selected negative examples from index
    negative_indices = shuffle(1:length(index.docs)) |>
                       x -> first(x, negatives_samples * length(concept.source_doc_ids))
    setdiff!(negative_indices, concept.source_doc_ids)
    # transpose to match the MLJLinearModels convention
    # let's duplicate positive to have more negative examples -- they are less "clean"
    X = vcat(concept.embeddings',
        @view(index.embeddings[:, negative_indices])')

    # expects labels as +-1
    y = vcat(ones(length(concept.source_doc_ids)),
        -1ones(length(negative_indices))) .|> Int

    ## Benchmark if it's a sensible model
    accuracy = cross_validate_accuracy(X, y; k = 4, verbose = false, lambda)
    verbose && @info "Cross-validated accuracy: $(round(accuracy;digits=2)*100)%"
    if accuracy <= 0.9
        @warn "Accuracy is too low! (Expected > 90%); revisit the regularization strength (smaller `lambda`), the `concept` wording (lens), or increase the sample size (`num_samples`)."
    end

    concept.coeffs = fit(model, X, y) .|> Float32
    # You can predict the score on the "concept" of a new document by multiplying coefficients with an embedding
    # y_pred = vec(X * coefficients) .|> sigmoid

    return concept
end

"""
    train_spectrum(index::AbstractDocumentIndex,
        spectrum::Tuple{String, String};
        num_samples::Int = 100, verbose::Bool = true,
        rewriter_template::Symbol = :StatementRewriter,
        lambda::Real = 1e-5,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple(),)

Train a Spectrum, ie, a two-sided axis of polar opposite concepts.

We effectively identify the "directions" in the embedding space that represent the two concepts that you selected as the opposite ends of the spectrum.

Practically, it takes a `num_samples` documents from `index`, rewrites them through the specified lenses (ends of spectrum), then embeds these rewritten documents, and finally trains a logistic regression model to classify the documents according to the spectrum.

See also: `train!`, `train_concept`, `score`

# Arguments
- `index::AbstractDocumentIndex`: An index containing the documents to be analyzed. This index should have been previously built using `build_index`.
- `spectrum::Tuple{String, String}`: A pair of strings representing the two lenses through which the documents will be rewritten. For example, ("optimistic", "pessimistic") could be a spectrum.
- `num_samples::Int` (optional): The number of documents to sample from the index for training. Defaults to 100.
- `verbose::Bool` (optional): If `true`, prints detailed logs during the process. Defaults to `true`.
- `rewriter_template::Symbol` (optional): The template used for rewriting statements. Defaults to `:StatementRewriter`.
- `lambda::Real` (optional): Regularization parameter for the logistic regression. Defaults to 1e-5. Adjust if your cross-validated accuracy is too low.
- `aigenerate_kwargs::NamedTuple` (optional): Additional arguments for the `aigenerate` function. See `?aigenerate` for more details.
- `aiembed_kwargs::NamedTuple` (optional): Additional arguments for the `aiembed` function. See `?aiembed` for more details.

# Returns
- A `TrainedSpectrum` object containing the trained model (`coeffs`), along with other relevant information like the rewritten document (`docs`) and embeddings (`embeddings`).

# Example
```julia
# Assuming `index` is an existing document index
my_spectrum = ("pessimistic", "optimistic")
spectrum = train_spectrum(index, my_spectrum)
```

Show the top 5 highest scoring documents for the spectrum 2 (`spectrum.spectrum[2]` which is "optimistic" in this example):
```julia
scores = score(index, spectrum)
index.docs[first(sortperm(scores, rev = true), 5)]

# Use rev=false to get the highest scoring documents for spectrum 1 (opposite end)
```

You can customize the analysis by passing additional arguments to the AI generation and embedding functions. For example, you can specify the model to use for generation and how many samples to use:

```julia
spectrum = train_spectrum(index,
    ("forward-looking", "dwelling in the past");
    num_samples = 50, aigenerate_kwargs = (; model = "gpt3t"))
```

This function utilizes large language models to rewrite and analyze the text, providing insights based on the specified spectrum. The output includes embeddings and a model capable of projecting new documents onto this spectrum for analysis.

For troubleshooting, you can fit the model manually and inspect the accuracy:

```julia
X = spectrum.embeddings'
# First half is spectrum 1, second half is spectrum 2
y = vcat(-1ones(length(spectrum.source_doc_ids)), ones(length(spectrum.source_doc_ids))) .|>
    Int
accuracy = cross_validate_accuracy(X, y; k = 4, lambda = 1e-8)
```

Or explore the source documents and re-written documents:
```julia
# source documents
index.docs[spectrum.source_doc_ids]

# re-written documents
spectrum.docs
```
"""
function train_spectrum(index::AbstractDocumentIndex,
        spectrum::Tuple{String, String};
        num_samples::Int = 100, verbose::Bool = true,
        rewriter_template::Symbol = :StatementRewriter,
        lambda::Real = 1e-5,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple(),)
    #
    source_doc_ids = shuffle(1:length(index.docs)) |> x -> first(x, num_samples)
    spectrum = TrainedSpectrum(; index_id = index.id,
        source_doc_ids,
        spectrum)
    train!(index, spectrum; verbose,
        rewriter_template,
        lambda,
        aigenerate_kwargs,
        aiembed_kwargs)
end

"""
    train!(index::AbstractDocumentIndex,
           spectrum::TrainedSpectrum;
           verbose::Bool = true,
           overwrite::Bool = false,
           rewriter_template::Symbol = :StatementRewriter,
           lambda::Real = 1e-5,
           aigenerate_kwargs::NamedTuple = NamedTuple(),
           aiembed_kwargs::NamedTuple = NamedTuple(),)

Finish a partially trained Spectrum or retrain an existing one (with `overwrite=true`).

See also: `train_spectrum`, `train_concept`, `score`

# Arguments
- `index::AbstractDocumentIndex`: The document index containing the documents to be analyzed.
- `spectrum::TrainedSpectrum`: The previously trained spectrum object to be trained.
- `verbose::Bool` (optional): If `true`, prints logs during the process. Defaults to `true`.
- `overwrite::Bool` (optional): If `true`, existing training data in the spectrum will be overwritten. Defaults to `false`.
- `rewriter_template::Symbol` (optional): The template used for rewriting statements. Defaults to `:StatementRewriter`.
- `lambda::Real` (optional): Regularization parameter for logistic regression. Defaults to 1e-5. Reduce if your cross-validated accuracy is too low.
- `aigenerate_kwargs::NamedTuple` (optional): Additional arguments for the `aigenerate` function. See `?aigenerate` for more details.
- `aiembed_kwargs::NamedTuple` (optional): Additional arguments for the `aiembed` function. See `?aiembed` for more details.

# Returns
- The updated `TrainedSpectrum` object containing the trained model (`coeffs`), along with other relevant information like the rewritten document (`docs`) and embeddings (`embeddings`).

# Example
```julia
# Assuming `index` and `spectrum` are pre-existing objects
trained_spectrum = train!(index, spectrum, verbose = true, overwrite = true)
```

This function allows for iterative improvement of a spectrum model, adapting to new data or refinements in the analysis framework.
"""
function train!(index::AbstractDocumentIndex,
        spectrum::TrainedSpectrum;
        verbose::Bool = true,
        overwrite::Bool = false,
        rewriter_template::Symbol = :StatementRewriter,
        lambda::Real = 1e-5,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple(),)
    @assert !isempty(spectrum.spectrum[1]) "Spectrum side #1 must be non-empty! (Provided: $(spectrum.spectrum[1]))"
    @assert !isempty(spectrum.spectrum[2]) "Spectrum side #2 must be non-empty! (Provided: $(spectrum.spectrum[2]))"
    @assert spectrum.spectrum[1]!=spectrum.spectrum[2] "Spectrum sides must be different! (Provided: $(spectrum.spectrum[1]), $(spectrum.spectrum[2]))"

    @assert index.id==spectrum.index_id "Index ID mismatch! (Provided Index: $(index.id), Expected: $(spectrum.index_id))"

    (; source_doc_ids) = spectrum
    cost_tracker = Threads.Atomic{Float64}(0.0)

    if isnothing(spectrum.docs) || overwrite
        ## Rewrite a few statements
        verbose && @info "Rewriting $(2*length(source_doc_ids)) documents..."
        cost_tracker = Threads.Atomic{Float64}(0.0)
        model = hasproperty(aigenerate_kwargs, :model) ? aigenerate_kwargs.model :
                PT.MODEL_CHAT
        spectrum1 = asyncmap(source_doc_ids) do i
            msg = aigenerate(rewriter_template; verbose = false,
                statement = index.docs[i],
                lens = spectrum.spectrum[1],
                aigenerate_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            replace(msg.content, "\"" => "") |> strip
        end
        spectrum2 = asyncmap(source_doc_ids) do i
            msg = aigenerate(rewriter_template; verbose = false,
                statement = index.docs[i],
                lens = spectrum.spectrum[2],
                aigenerate_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            replace(msg.content, "\"" => "") |> strip
        end

        spectrum.docs = vcat(spectrum1, spectrum2)
    end
    @assert length(spectrum.docs)==2 * length(source_doc_ids) "Number of documents mismatch! (Provided: $(length(spectrum.docs)), Expected: $(2*length(source_doc_ids)))"

    if isnothing(spectrum.embeddings) || overwrite
        ## Embed them // assumes consistency, ie, documents have not changed
        verbose && @info "Embedding $(2*length(source_doc_ids)) documents..."
        model = hasproperty(aiembed_kwargs, :model) ? aiembed_kwargs.model :
                PT.MODEL_EMBEDDING
        @views spectrum1 = spectrum.docs[begin:length(source_doc_ids)]
        spectrum1_embeddings = let
            msg = aiembed(spectrum1, normalize; verbose = false, aiembed_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            msg.content
        end
        @views spectrum2 = spectrum.docs[(length(source_doc_ids) + 1):end]
        spectrum2_embeddings = let
            msg = aiembed(spectrum2, normalize; verbose = false, aiembed_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            msg.content
        end
        # We remove the embedding of the original source documents 
        # to retain only the "direction" of the lens/spectrum we chose
        embeddings = hcat(spectrum1_embeddings .-
                          @view(index.embeddings[:, source_doc_ids]),
            spectrum2_embeddings .- @view(index.embeddings[:, source_doc_ids])) .|> Float32
        spectrum.embeddings = embeddings
    end
    @assert size(spectrum.embeddings, 2)==2 * length(source_doc_ids) "Number of embeddings mismatch! (Provided: $(size(spectrum.embeddings, 2)), Expected: $(2*length(source_doc_ids)))"

    verbose && cost_tracker[] > 0 &&
        @info "Done with LLMs. Total cost: \$$(round(cost_tracker[],digits=3))"

    ## Train a classifier // always retrain, it's super fast
    verbose && @info "Training a classifier..."
    model = LogisticRegression(lambda; fit_intercept = false)
    # transpose to match the MLJLinearModels convention
    X = spectrum.embeddings'
    # expects labels as +-1
    y = vcat(-1ones(length(source_doc_ids)), ones(length(source_doc_ids))) .|> Int

    ## Benchmark if it's a sensible model
    accuracy = cross_validate_accuracy(X, y; k = 4, verbose = false, lambda)
    verbose && @info "Cross-validated accuracy: $(round(accuracy;digits=2)*100)%"
    if accuracy <= 0.9
        @warn "Accuracy is too low! (Expected > 90%); revisit the regularization strength (smaller `lambda`), the `spectrum` (lenses), or increase the sample size (`num_samples`)."
    end

    spectrum.coeffs = fit(model, X, y) .|> Float32
    # You can predict the score on the "spectrum" of a new document by multiplying coefficients with an embedding
    # y_pred = vec(X * coefficients) .|> sigmoid

    return spectrum
end

"""
    score(index::AbstractDocumentIndex,
        spectrum::TrainedSpectrum;
        check_index::Bool = true)

Scores all documents in the provided `index` based on the `TrainedSpectrum`. 

The score reflects how closely each document aligns with each of the ends of the trained spectrum. 
Scores are left-to-right, ie, a score closer to 0 indicates a higher alignment to `spectrum.spectrum[1]` and a score closer to 1 indicates a higher alignment to `spectrum.spectrum[2]`.

# Arguments
- `index::AbstractDocumentIndex`: The index containing the documents to be scored.
- `spectrum::TrainedSpectrum`: The trained spectrum model used for scoring.
- `check_index::Bool` (optional): If `true`, checks for index ID matching between the provided index and the one used in the spectrum training. Defaults to `true`.

# Returns
- A vector of scores, each corresponding to a document in the index, in the range [0, 1].

# Example
```julia
# Assuming `index` and `spectrum` are predefined
scores = score(index, spectrum)
```

You can show the top 5 highest scoring documents for the spectrum 2:
```julia
index.docs[first(sortperm(scores, rev = true), 5)]

# Use rev=false if you want to see documents closest to spectrum 1 (opposite end)
```

This function is useful for ranking all documents along the chosen `spectrum`.
"""
function score(index::AbstractDocumentIndex,
        spectrum::TrainedSpectrum;
        check_index::Bool = true)
    # Check if the spectrum is properly trained
    @assert !isempty(spectrum.coeffs) "The spectrum is not trained. Coefficients are missing. Use `train!`."
    if check_index && index.id != spectrum.index_id
        @warn "Potential error: Index ID mismatch! (Provided Index: $(index.id), Used for training: $(spectrum.index_id))"
    end
    # scores between 0 and 1
    return vec(spectrum.coeffs' * index.embeddings) .|> sigmoid
end

"""
    (spectrum::TrainedSpectrum)(index::AbstractDocumentIndex; check_index::Bool = true)

A method definition that allows a `TrainedSpectrum` object to be called as a function to score documents in an `index`. This method delegates to the `score` function.

The score reflects how closely each document aligns with each of the ends of the trained spectrum. 
Scores are left-to-right, ie, a score closer to 0 indicates a higher alignment to `spectrum.spectrum[1]` and a score closer to 1 indicates a higher alignment to `spectrum.spectrum[2]`.

# Arguments
- `index::AbstractDocumentIndex`: The index containing the documents to be scored.
- `check_index::Bool` (optional): If `true`, performs a check to ensure that the index ID matches the one used in the spectrum training. Defaults to `true`.

# Returns
- A vector of scores in the range [0, 1], each corresponding to a document in the index.

# Example
```julia
# Assuming `index` and `spectrum` are predefined
scores = spectrum(index)
```

This method provides a convenient and intuitive way to apply a trained spectrum model to a document index for scoring.
"""
function (spectrum::TrainedSpectrum)(index::AbstractDocumentIndex; check_index::Bool = true)
    return score(index, spectrum; check_index)
end

"""
    score(index::AbstractDocumentIndex, concept::TrainedConcept; check_index::Bool = true)

Scores all documents in the provided `index` based on the `TrainedConcept`. 

The score quantifies the relevance or alignment of each document with the trained concept, with a score closer to 1 indicating a higher relevance.

The function uses a sigmoid function to map the scores to a range between 0 and 1, providing a probability-like interpretation.

# Arguments
- `index::AbstractDocumentIndex`: The index containing the documents to be scored.
- `concept::TrainedConcept`: The trained concept model used for scoring.
- `check_index::Bool` (optional): If `true`, checks for index ID matching between the provided index and the one used in the concept training. Defaults to `true`.

# Returns
- A vector of scores, each corresponding to a document in the index, in the range [0, 1].

# Example
```julia
# Assuming `index` and `concept` are predefined
scores = score(index, concept)
```

You can show the top 5 highest scoring documents for the concept:
```julia
index.docs[first(sortperm(scores, rev = true), 5)]
```

This function is particularly useful for analyzing the presence, intensity, or relevance of a specific concept within a collection of documents.
"""
function score(index::AbstractDocumentIndex,
        concept::TrainedConcept;
        check_index::Bool = true)
    # Check if the concept is properly trained
    @assert !isempty(concept.coeffs) "The concept is not trained. Coefficients are missing. Use `train!`."
    if check_index && index.id != concept.index_id
        @warn "Potential error: Index ID mismatch! (Provided Index: $(index.id), Used for training: $(concept.index_id))"
    end
    # scores between 0 and 1
    return vec(concept.coeffs' * index.embeddings) .|> sigmoid
end

"""
    (concept::TrainedConcept)(index::AbstractDocumentIndex; check_index::Bool = true)

A method definition that allows a `TrainedConcept` object to be called as a function to score documents in an `index`. This method delegates to the `score` function.

# Arguments
- `index::AbstractDocumentIndex`: The index containing the documents to be scored.
- `check_index::Bool` (optional): If `true`, performs a check to ensure that the index ID matches the one used in the concept training. Defaults to `true`.

# Returns
- A vector of scores in the range [0, 1], each corresponding to a document in the index.

# Example
```julia
# Assuming `index` and `concept` are predefined
scores = concept(index)
```

This method provides a convenient and intuitive way to apply a trained concept model to a document index for scoring, facilitating thematic analysis and concept relevance studies.
"""
function (concept::TrainedConcept)(index::AbstractDocumentIndex; check_index::Bool = true)
    return score(index, concept; check_index)
end