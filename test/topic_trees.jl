using LLMTextAnalysis: TopicTreeNode, TopicMetadata, topic_tree, topic_tree!, print_tree

@testset "TopicTreeNode" begin
    # Test TopicTreeNode construction
    topic = TopicMetadata(index_id = :test, docs_idx = 1:100, topic_level = 1,
        topic_idx = 1, label = "Test Topic")
    node = TopicTreeNode(topic = topic, total_docs = 1000)
    @test node isa TopicTreeNode
    @test node.topic == topic
    @test node.total_docs == 1000
    @test isempty(node.children)

    # Test Base.IteratorEltype and Base.eltype
    @test Base.IteratorEltype(TreeIterator{TopicTreeNode}) == Base.HasEltype()
    @test Base.eltype(TreeIterator{TopicTreeNode}) == TopicTreeNode

    # Test AbstractTrees.childtype
    @test AbstractTrees.childtype(TopicTreeNode) == TopicTreeNode

    # Test AbstractTrees.nodevalue
    topic = TopicMetadata(index_id = :test, docs_idx = 1:50, topic_level = 2,
        topic_idx = 3, label = "Test Topic")
    node = TopicTreeNode(topic = topic, total_docs = 100)
    value = AbstractTrees.nodevalue(node)
    @test value == "Test Topic (N: 50, Share: 50.0%, Level: 2, Topic ID: 3)"

    # Test AbstractTrees.children
    parent = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 1:100,
            topic_level = 1, topic_idx = 1, label = "Parent"),
        total_docs = 100)
    child1 = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 1:50,
            topic_level = 2, topic_idx = 2, label = "Child1"),
        total_docs = 100)
    child2 = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 51:100,
            topic_level = 2, topic_idx = 3, label = "Child2"),
        total_docs = 100)
    push!(parent.children, child1, child2)
    @test AbstractTrees.children(parent) == [child1, child2]

    # Test Base.show
    topic = TopicMetadata(index_id = :test, docs_idx = 1:75, topic_level = 3,
        topic_idx = 4, label = "Show Test")
    node = TopicTreeNode(topic = topic, total_docs = 150)
    io = IOBuffer()
    show(io, node)
    @test String(take!(io)) == "Show Test (N: 75, Share: 50.0%, Level: 3, Topic ID: 4)"

    # Test complex scenario with nested children
    root = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 1:100,
            topic_level = 0, topic_idx = 0, label = "Root"),
        total_docs = 100)
    child1 = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 1:60,
            topic_level = 1, topic_idx = 1, label = "Child1"),
        total_docs = 100)
    child2 = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 61:100,
            topic_level = 1, topic_idx = 2, label = "Child2"),
        total_docs = 100)
    grandchild1 = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 1:30, topic_level = 2,
            topic_idx = 3, label = "Grandchild1"),
        total_docs = 100)
    grandchild2 = TopicTreeNode(
        topic = TopicMetadata(index_id = :test, docs_idx = 31:60, topic_level = 2,
            topic_idx = 4, label = "Grandchild2"),
        total_docs = 100)

    push!(root.children, child1, child2)
    push!(child1.children, grandchild1, grandchild2)

    @test AbstractTrees.children(root) == [child1, child2]
    @test AbstractTrees.children(child1) == [grandchild1, grandchild2]
    @test AbstractTrees.children(child2) == TopicTreeNode[]
    @test AbstractTrees.nodevalue(root) ==
          "Root (N: 100, Share: 100.0%, Level: 0, Topic ID: 0)"
    @test AbstractTrees.nodevalue(child1) ==
          "Child1 (N: 60, Share: 60.0%, Level: 1, Topic ID: 1)"
    @test AbstractTrees.nodevalue(grandchild1) ==
          "Grandchild1 (N: 30, Share: 30.0%, Level: 2, Topic ID: 3)"
end

@testset "topic_tree" begin
    # Helper function to create a mock DocIndex
    function create_mock_index(levels)
        docs = ["doc$i" for i in 1:10]
        topic_levels = Dict{Union{Int, AbstractString}, Vector{TopicMetadata}}()
        for level in levels
            topic_levels[level] = [
                TopicMetadata(index_id = :test, docs_idx = 1:5, topic_level = level,
                    topic_idx = 1, label = "Topic1", center_doc_idx = 1),
                TopicMetadata(index_id = :test, docs_idx = 6:10, topic_level = level,
                    topic_idx = 2, label = "Topic2", center_doc_idx = 6)
            ]
        end
        return DocIndex(;
            id = :test, docs, topic_levels, embeddings = Matrix{Float32}(undef, 10, 10),
            distances = Matrix{Float32}(undef, 10, 10))
    end

    # Test 1: Basic functionality - single level
    index = create_mock_index([4])
    root = topic_tree(index, [4])

    @test root.total_docs == 10
    @test root.topic.label == "All Documents"
    @test length(root.children) == 2
    @test all(child.topic.topic_level == 4 for child in root.children)

    # Test 2: Multiple levels
    index = create_mock_index([4, 10])
    root = topic_tree(index, [4, 10])

    @test length(root.children) == 2
    @test all(length(child.children) == 1 for child in root.children)
    @test all(grandchild.topic.topic_level == 10 for child in root.children
    for grandchild in child.children)

    # Test 3: Sorted vs Unsorted
    index = create_mock_index([4])
    root_sorted = topic_tree(index, [4], sorted = true)
    root_unsorted = topic_tree(index, [4], sorted = false)

    @test length(root_sorted.children) == length(root_unsorted.children)
    @test root_sorted.children != root_unsorted.children  # Order should be different

    # Test 4: Error handling - non-existent level
    index = create_mock_index([4])
    @test_throws AssertionError topic_tree(index, [5])

    # Test 6: Large number of levels
    levels = collect(1:100)
    index = create_mock_index(levels)
    root = topic_tree(index, levels)

    @test length(root.children) == 2

    # Check if the tree has the correct depth
    function tree_depth(node)
        isempty(node.children) ? 0 :
        1 + maximum(tree_depth(child) for child in node.children)
    end

    @test tree_depth(root) == length(levels)

    # test printing
    index = create_mock_index([4, 10])
    root = topic_tree(index, [4, 10])
    io = IOBuffer()
    print_tree(io, root)
    output = String(take!(io))
    @test occursin("All Documents (N: 10, Share: 100.0%, Level: root, Topic ID: 0)", output)
    @test occursin("Topic1 (N: 5, Share: 50.0%, Level: 4, Topic ID: 1)", output)
    @test occursin("Topic2 (N: 5, Share: 50.0%, Level: 4, Topic ID: 2)", output)
end
