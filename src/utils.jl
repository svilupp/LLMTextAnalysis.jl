"Counts number of unique elements in a vector"
nunique(vect::AbstractVector) = length(unique(vect))
sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))