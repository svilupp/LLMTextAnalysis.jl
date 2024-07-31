"""
    TopicTreeNode

A node in the topic hierarchy of the `index`

# Fields
- `topic::TopicMetadata`: The metadata of the topic
- `total_docs::Int`: The total number of documents in the `index`
- `children::Vector{TopicTreeNode}`: Children nodes
"""
@kwdef mutable struct TopicTreeNode
    topic::TopicMetadata
    total_docs::Int
    children::Vector{TopicTreeNode} = TopicTreeNode[]
end

Base.IteratorEltype(::Type{<:TreeIterator{TopicTreeNode}}) = Base.HasEltype()
function Base.eltype(::Type{<:TreeIterator{T}}) where {T <: TopicTreeNode}
    T
end
function AbstractTrees.childtype(::Type{T}) where {T <: TopicTreeNode}
    T
end
function AbstractTrees.nodevalue(n::TopicTreeNode)
    (; label, topic_idx, docs_idx, topic_level) = n.topic
    N = length(docs_idx)
    string(label,
        " (N: $(N), Share: $(round(100*N/n.total_docs; digits=2))%, Level: $(topic_level), Topic ID: $(topic_idx))")
end
function AbstractTrees.children(node::TopicTreeNode)
    return node.children
end
AbstractTrees.parent(n::TopicTreeNode) = n.parent
function Base.show(io::IO, node::TopicTreeNode)
    str = AbstractTrees.nodevalue(node)
    print(io, str)
end

"""
    topic_tree(
        index::DocIndex, levels::AbstractVector{<:Union{Integer, AbstractString}};
        sorted::Bool = true)

Builds a topic tree from the `index` for the provided `levels`.
 Levels must be present in the index, eg, run `build_cluster!` first.

# Arguments
- `index::DocIndex`: The document index
- `levels::AbstractVector{<:Union{Integer, AbstractString}}`: The levels to include in the tree, eg, `[4, 10, 20]` (they must be present in index.topic_levels)
- `sorted::Bool`: Whether to sort the children by the number of documents in each topic. Defaults to `true`.

# Example
```julia

# Create topic tree for levels k=4, k=10, k=20
root = topic_tree(index, [4, 10, 20])

# Display it
print_tree(root)
# example output
# "All Documents (N: 10, Share: 100.0%, Level: root, Topic ID: 0)"
# ├─ "Topic1 (N: 5, Share: 50.0%, Level: 4, Topic ID: 1)"
# │  └─ "Topic1 (N: 5, Share: 50.0%, Level: 10, Topic ID: 1)"
# ...
# └─ "Topic2 (N: 5, Share: 50.0%, Level: 4, Topic ID: 2)"
#    └─ "Topic2 (N: 5, Share: 50.0%, Level: 10, Topic ID: 2)"
# ...
```
"""
function topic_tree(
        index::DocIndex, levels::AbstractVector{<:Union{Integer, AbstractString}};
        sorted::Bool = true)
    ## Checks
    for level in levels
        @assert haskey(index.topic_levels, level) "Clustering topic level $level not found in index - run `build_cluster!` first."
    end
    root_topic = TopicMetadata(; index_id = index.id, docs_idx = 1:length(index.docs),
        topic_level = "root", topic_idx = 0, label = "All Documents")
    root = TopicTreeNode(; topic = root_topic, total_docs = length(index.docs))
    nodes_current = TopicTreeNode[root]
    nodes_to_scan = TopicTreeNode[]
    for level in levels
        topics = index.topic_levels[level]
        for node in nodes_current
            ## find linkage between provided node and the nodes on `level`
            topic_tree!(node, topics; sorted)
            append!(nodes_to_scan, AbstractTrees.children(node))
        end
        nodes_current = nodes_to_scan
        nodes_to_scan = TopicTreeNode[]
    end
    return root
end

# Inner method for individual nodes
function topic_tree!(
        parent::TopicTreeNode, topics::AbstractVector{<:TopicMetadata}; sorted::Bool = true)
    parent_docs_idx = parent.topic.docs_idx
    ## Get the topics at the current level
    for topic in topics
        ## find the corresponding children
        if topic.center_doc_idx in parent_docs_idx
            push!(parent.children, TopicTreeNode(; topic, total_docs = parent.total_docs))
        end
    end
    if sorted
        sort!(parent.children, by = x -> length(x.topic.docs_idx), rev = true)
    end
    return parent
end