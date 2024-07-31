"Counts number of unique elements in a vector"
nunique(vect::AbstractVector) = length(unique(vect))
sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))
function softmax(x::AbstractVector)
    exp_ = exp.(x)
    exp_ / sum(exp_)
end
function softmax(x::AbstractMatrix)
    temp = exp.(x)
    sum_ = sum(temp, dims = 2) |> vec
    for j in axes(temp, 2)
        # divide by sum of each row
        temp[:, j] ./= sum_
    end
    return temp
end