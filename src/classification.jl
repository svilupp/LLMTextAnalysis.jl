
"""
    train_classifier(index::AbstractDocumentIndex,
        labels::AbstractVector{<:AbstractString};
        docs_ids::AbstractVector{<:Integer} = Int[],
        docs_labels::AbstractVector{<:Integer} = Int[],
        labels_description::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        num_samples::Int = 5, verbose::Bool = true,
        writer_template::Symbol = :TextWriterFromLabel,
        lambda::Real = 1e-3,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple())

Train a model to classify each document into one of several specific topics based on `labels` (detailed in `labels_description`).

If user provides documents from the index and their corresponding labels (`docs_ids` and `docs_labels`, respectively), 
the model will be trained on those documents. Aim for a balanced dataset (all `labels` must be present) and a minimum of 5 documents per label (ideally more).

Otherwise, we will first generate `num_samples` of synthetic documents for each label in `labels`, ie, in total there will be `num_samples x length(labels)` generated documents.
If `labels_description` is provided, these descriptions will be provided to the AI model to generate more diverse and relevant documents for each label (more informative than just one word labels).

Under the hood, we train a multi-label classifier on top of the embeddings of the documents.

The resulting scores will be a matrix of probabilities for each document and each label.
Scores dimension: `num_documents x num_labels`, ie, position [1,3] would correspond to the probability of the first document corresponding to the 3rd label/class.
To pick the best label for each document, you can use `argmax(scores, dims=2)`.

See also: `score`, `train!`, `train_spectrum`, `train_concept`

# Arguments
- `index::AbstractDocumentIndex`: An index containing the documents to be analyzed.
- `labels::AbstractVector{<:AbstractString}`: A vector of labels to be used for training the classifier (documents will be assigned to one of these labels).
- `docs_ids::AbstractVector{<:Integer}` (optional): The IDs of the documents in the `index` to be used for training. Defaults to an empty vector and will generate synthetic documents.
- `docs_labels::AbstractVector{<:Integer}` (optional): The labels corresponding to the documents in `docs_ids`. Defaults to an empty vector.
- `labels_description::Union{Nothing, AbstractVector{<:AbstractString}}` (optional): A vector of descriptions for each label. If provided, it will be used to generate more diverse and relevant documents for each label. Defaults to `nothing`.
- `num_samples::Int` (optional): The number of documents to generate for each label in `labels`. Defaults to 5.
- `verbose::Bool` (optional): If `true`, prints detailed logs during the process. Defaults to `true`.
- `writer_template::Symbol` (optional): The template used for writing synthetic documents. Defaults to `:TextWriterFromLabel`.
- `lambda::Real` (optional): Regularization parameter for logistic regression. Defaults to 1e-3
- `aigenerate_kwargs::NamedTuple` (optional): Additional arguments for the `aigenerate` function. See `?aigenerate` for more details.
- `aiembed_kwargs::NamedTuple` (optional): Additional arguments for the `aiembed` function. See `?aiembed` for more details.

# Returns
- A `TrainedClassifier` object containing the trained model, along with relevant information such as the generated documents (`docs`), embeddings (`embeddings`), and model coefficients (`coeffs`).

# Example

Create a classifier for a set of labeled documents in our index (ie, we know the labels for some documents):
```julia
# Assuming `index` is an existing document index
```


"""
function train_classifier(index::AbstractDocumentIndex,
        labels::AbstractVector{<:AbstractString};
        docs_ids::AbstractVector{<:Integer} = Int[],
        docs_labels::AbstractVector{<:Integer} = Int[],
        labels_description::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        num_samples::Int = 5, verbose::Bool = true,
        writer_template::Symbol = :TextWriterFromLabel,
        lambda::Real = 1e-3,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple())
    ## Checks
    @assert nunique(labels)>1 "At least two different labels are required! (Provided: $(unique(labels)))"
    @assert length(docs_ids)==length(docs_labels) "Number of documents and labels mismatch! (Provided: $(length(docs_ids)), $(length(docs_labels)))"
    if !isempty(docs_ids)
        ## check consistency
        @assert nunique(docs_labels)==nunique(labels) "Number of unique labels and document labels mismatch! (Provided: $(nunique(labels)), $(nunique(docs_labels)))"
        @assert all(1 .<= docs_labels .<= nunique(labels)) "Document labels must be within the 1 to number of unique labels! (Provided: $(extrema(docs_labels)))"
        @assert all(1 .<= docs_ids .<= length(index.docs)) "Document IDs must be within the 1 to number of documents! (Provided: $(extrema(docs_ids)))"
    end
    if !isnothing(labels_description)
        @assert length(labels)==length(labels_description) "Number of labels and their descriptions mismatch! (Provided: $(length(labels)), $(length(labels_description)))"
    end

    classifier = TrainedClassifier(; index_id = index.id,
        source_doc_ids = isempty(docs_ids) ? nothing : docs_ids,
        ## Load the documents and embeddings if provided
        docs = isempty(docs_ids) ? nothing : @view(index.docs[docs_ids]),
        embeddings = isempty(docs_ids) ? nothing : @view(index.embeddings[:, docs_ids]),
        docs_labels,
        labels,
        labels_description)
    train!(index, classifier; verbose,
        writer_template,
        lambda, num_samples,
        aigenerate_kwargs,
        aiembed_kwargs)
end

"""
    train!(index::AbstractDocumentIndex,
        classifier::TrainedClassifier;
        verbose::Bool = true,
        overwrite::Bool = false,
        writer_template::Symbol = :TextWriterFromLabel,
        lambda::Real = 1e-3, num_samples::Int = 5,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple())

Refine or retrain a previously trained `TrainedClassifier` model. 

This function can be used to update the classifier model with new data, adjust parameters, or completely retrain it.

See also: `train_classifier`, `score`

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
        classifier::TrainedClassifier;
        verbose::Bool = true,
        overwrite::Bool = false,
        writer_template::Symbol = :TextWriterFromLabel,
        lambda::Real = 1e-3, num_samples::Int = 5,
        aigenerate_kwargs::NamedTuple = NamedTuple(),
        aiembed_kwargs::NamedTuple = NamedTuple())
    ## Checks
    @assert !isempty(classifier.labels) "Labels must be non-empty! (Provided: $(classifier.labels))"
    @assert index.id==classifier.index_id "Index ID mismatch! (Provided Index: $(index.id), Expected: $(classifier.index_id))"

    cost_tracker = Threads.Atomic{Float64}(0.0)

    ## Fill docs_labels if not provided - `num_samples` times each label
    if isempty(classifier.docs_labels)
        classifier.docs_labels = repeat(1:length(classifier.labels), inner = num_samples) |>
                                 shuffle
    end

    if isnothing(classifier.docs) || overwrite
        ## Rewrite a few statements
        verbose && @info "Generating $(length(classifier.docs_labels)) documents..."
        cost_tracker = Threads.Atomic{Float64}(0.0)
        model = hasproperty(aigenerate_kwargs, :model) ? aigenerate_kwargs.model :
                PT.MODEL_CHAT
        ## Decide whether to generate based on labels or descriptions
        labels_to_use = isnothing(classifier.labels_description) ? classifier.labels :
                        classifier.labels_description
        classifier.docs = asyncmap(classifier.docs_labels) do i
            msg = aigenerate(writer_template; verbose = false,
                label = labels_to_use[i],
                ## Grab random document as a sample reference to match (for diversity)
                sample = rand(index.docs),
                aigenerate_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            replace(msg.content, "\"" => "") |> strip
        end
    end
    @assert length(classifier.docs)==length(classifier.docs_labels) "Number of documents and their labels do not match! (Provided: $(length(classifier.docs)), Expected: $(length(classifier.docs_labels)))"

    if isnothing(classifier.embeddings) || overwrite
        ## Embed them // assumes consistency, ie, documents have not changed
        verbose && @info "Embedding $(length(classifier.docs_labels)) documents..."
        model = hasproperty(aiembed_kwargs, :model) ? aiembed_kwargs.model :
                PT.MODEL_EMBEDDING
        embeddings = let
            msg = aiembed(classifier.docs, normalize; verbose = false, aiembed_kwargs...)
            Threads.atomic_add!(cost_tracker, PT.call_cost(msg, model)) # track costs
            msg.content
        end
        classifier.embeddings = embeddings .|> Float32
    end
    @assert size(classifier.embeddings, 2)==length(classifier.docs_labels) "Number of embeddings mismatch! (Provided: $(size(classifier.embeddings, 2)), Expected: $(length(classifier.docs_labels)))"

    verbose && cost_tracker[] > 0 &&
        @info "Done with LLMs. Total cost: \$$(round(cost_tracker[],digits=3))"

    ## Train a classifier // always retrain, it's super fast
    verbose && @info "Training a classifier..."
    model = MultinomialRegression(lambda; fit_intercept = false)
    # transpose to match the MLJLinearModels convention
    X = classifier.embeddings'

    # expects labels (integers)
    y = classifier.docs_labels

    ## TODO: add a cross-validation check
    ## Benchmark if it's a sensible model
    ## accuracy = cross_validate_accuracy(X, y; k = 4, verbose = false, lambda)
    ## verbose && @info "Cross-validated accuracy: $(round(accuracy;digits=2)*100)%"
    ## if accuracy <= 0.9
    ##     @warn "Accuracy is too low! (Expected > 90%); revisit the regularization strength (smaller `lambda`), or increase the sample size (`num_samples`) or provide more labels (`docs_ids`, `docs_labels`)!"
    ## end

    classifier.coeffs = fit(model, X, y) |> x -> reshape(x, size(X, 2), :) .|> Float32
    # You can predict the score on the "classifier" of a new document by multiplying coefficients with an embedding
    # y_pred = softmax(X * coefficients) # matrix of scores for (document x label)

    return classifier
end

"""
    score(index::AbstractDocumentIndex,
        classifier::TrainedClassifier;
        check_index::Bool = true)

Scores all documents in the provided `index` based on the `TrainedClassifier`. 

The score reflects how closely each document aligns to each label in the trained classifier (`classifier.labels`). 

The resulting scores will be a matrix of probabilities for each document and each label.

Scores dimension: `num_documents x num_labels`, ie, position [1,3] would correspond to the probability of the first document corresponding to the 3rd label/class.

To pick the best label for each document, you can use `argmax(scores, dims=2)`.

# Arguments
- `index::AbstractDocumentIndex`: The index containing the documents to be scored.
- `classifier::TrainedClassifier`: The trained classifier model used for scoring.
- `check_index::Bool` (optional): If `true`, checks for index ID matching between the provided index and the one used in the classifier training. Defaults to `true`.

# Returns
- A matrix of scores, each row corresponding to a document in the index and each column corresponding to probability of that label.

# Example
```julia
# Assuming `index` and `classifier` are predefined
scores = score(index, classifier)
```

Pick the highest scoring label for each document:
```julia
scores = score(index, classifier)
label_ids = argmax(scores, dims = 2) |> vec |> x -> map(i -> i[2], x)
best_labels = classifier.labels[label_ids]
```
"""
function score(index::AbstractDocumentIndex,
        classifier::TrainedClassifier;
        check_index::Bool = true)
    # Check if the classifier is properly trained
    @assert !isempty(classifier.coeffs) "The classifier is not trained. Coefficients are missing. Use `train!`."
    if check_index && index.id != classifier.index_id
        @warn "Potential error: Index ID mismatch! (Provided Index: $(index.id), Used for training: $(classifier.index_id))"
    end
    # scores between 0 and 1
    return softmax(index.embeddings' * classifier.coeffs)
end

"""
    (classifier::TrainedClassifier)(
        index::AbstractDocumentIndex; check_index::Bool = true)

A method definition that allows a `TrainedClassifier` object to be called as a function to score documents in an `index`. This method delegates to the `score` function.

The score reflects how closely each document aligns to each label in the trained classifier (`classifier.labels`). 

The resulting scores will be a matrix of probabilities for each document and each label.

Scores dimension: `num_documents x num_labels`, ie, position [1,3] would correspond to the probability of the first document corresponding to the 3rd label/class.

To pick the best label for each document, you can use `argmax(scores, dims=2)`.

# Arguments
- `index::AbstractDocumentIndex`: The index containing the documents to be scored.
- `check_index::Bool` (optional): If `true`, performs a check to ensure that the index ID matches the one used in the classifier training. Defaults to `true`.

# Returns
- A vector of scores in the range [0, 1], each corresponding to a document in the index.

# Example
```julia
# Assuming `index` and `classifier` are predefined
scores = classifier(index)
```

Pick the highest scoring label for each document:
```julia
scores = score(index, classifier)
label_ids = argmax(scores, dims = 2) |> vec |> x -> map(i -> i[2], x)
best_labels = classifier.labels[label_ids]
```
This method provides a convenient and intuitive way to apply a trained classifier model to a document index for scoring.
"""
function (classifier::TrainedClassifier)(
        index::AbstractDocumentIndex; check_index::Bool = true)
    return score(index, classifier; check_index)
end