"Counts number of unique elements in a vector"
nunique(vect::AbstractVector) = length(unique(vect))
sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))

"""
    wrap_string(str::String,
        text_width::Int = 20;
        newline::Union{AbstractString, AbstractChar} = '\n')

Breaks a string into lines of a given `text_width`.
Optionally, you can specify the `newline` character or string to use.

# Example:

```julia
wrap_string("Certainly, here's a function in Julia that will wrap a string according to the specifications:", 10) |> print
```
"""
function wrap_string(str::String,
        text_width::Int = 20;
        newline::Union{AbstractString, AbstractChar} = '\n')
    words = split(str)
    output = IOBuffer()
    current_line_length = 0

    for word in words
        word_length = length(word)
        if current_line_length + word_length > text_width
            if current_line_length > 0
                write(output, newline)
                current_line_length = 0
            end
            while word_length > text_width
                write(output, word[1:(text_width - 1)], "-$newline")
                word = word[text_width:end]
                word_length -= text_width - 1
            end
        end
        if current_line_length > 0
            write(output, ' ')
            current_line_length += 1
        end
        write(output, word)
        current_line_length += word_length
    end

    return String(take!(output))
end;
