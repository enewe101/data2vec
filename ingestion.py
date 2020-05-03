import scipy
import os
import numpy
from collections import Counter, defaultdict
import itertools as it
import numbers
import d2v
import scipy.sparse



def ingest(object_iterator, path):
    """
    Record the data in an internal datastructure that is fit for the purpose
    of counting interactions and measuring relationships.

    Use a string-based namespace to record the graph of interactions.
    Maintain a dictionary that maps these string IDs to integers so that at
    any given time we can operate on the graph in numpy.

    Inputs:

     - object_iterator - iterator<dict> - An iterator that yields dictionaaries
        (as if json-loaded) which is the basic way of indicating interactions
        between objects.  Each object iterated is taken to be a non-primitive
        object whose contents expresses relationships among primitive or
        non-primitive parts.

     - path - str - directory in which to record a model (see returned
       `graph`) and two dictionaries (mappings of string IDs in
       `graph` to integer IDs.

    Returns:
    
     - (graph, dictionary)

        graph - dict<list<str>> - for each object defined by the
            object iterator, it records a list of IDs corresponding to 
            its contents.  These IDs refer either to primitive objects or
            to non_primitive objcts (which must be done by reference).

        dictionary - Dictionary - A mapping between string IDs (as appear in 
            `graph` and integer IDs, for all objects, whether primitive
            or not, in 

    """

    # Make sure we can write to path.
    if path is not None and not os.path.exists(path):
        os.makedirs(path)
        test_write(path)

    # `graph` records the structure in each object.
    dictionary = d2v.dictionary.Dictionary()
    graph = {}

    # Assign int IDs to all values in obj_iterator; record structure as graph.
    for obj in object_iterator:
        obj_id = dictionary.add(d2v.d2v_id.get_non_primitive_id(obj))
        child_ids = dictionary.add_many(d2v.d2v_id.get_child_ids(obj))
        graph[obj_id] = child_ids

    # Convert graph to CSR sparse matrix
    shape = (len(dictionary), len(dictionary))
    graph_adjacency = d2v.graph.graph_to_csr(graph, shape=shape, dtype=bool)

    # Create a pairlist and an expanded graph; 
    pairs, expanded_graph = make_pairs_and_expanded_graph(graph)

    # Convert both to CSR sparse matrix
    expanded_graph_adjacency = d2v.graph.graph_to_csr(
        expanded_graph, shape=shape, dtype=int)
    pairs_adjacency = d2v.pairlist.pairlist_to_coo(
        Counter(pairs), shape=shape, symmetric=True
    )

    # If we don't need to write then we're done, return the results.
    if path is None:
        return graph, dictionary

    # Save dictionary.
    d2v.dictionary.write_dictionary(
        os.path.join(path, 'dictionary.txt'),
        dictionary
    )

    # Save graph as adjacency matrix
    scipy.sparse.save_npz(os.path.join(path, 'graph.npz'), graph_adjacency)

    # Save pairlist as edgelist and adjacency matrix.
    d2v.pairlist.write_pairlist(os.path.join(path, 'pairs.tsv'), pairs)
    scipy.sparse.save_npz(
        os.path.join(path, 'pairs.npz'),
        pairs_adjacency
    )

    # Save expanded graph
    scipy.sparse.save_npz(
        os.path.join(path, 'expanded-graph.npz'),
        expanded_graph_adjacency
    )

    return graph, dictionary



def test_write(path):
    test_path = os.path.join(path, 'test') 
    with open(test_path) as test_write:
        test_write.write('test')
    os.remove(test_path)


def make_pairs_and_expanded_graph(graph):
    """
    Operates on a dictionary, `graph` whose keys are the indices for
    non-primitives and whose values are lists containing the indices of the
    corresponding children.  This function iterates over the items in `graph`,
    and generates two derivative data structures, a pair list `pairs`, and a an
    `expanded_graph`.

    Inputs
     - `graph` - dict<list<str>> - dictionary that maps non-primitives to their
       children.  The keys are non_primitive indices and the values are lists
       of the corresponding child indices.

    Outputs
     - `pairs` - list<tuple<str>> - lists all pairs of children (by d2v_id)
        that cooccur within a non-primitive.  This is calculated after
        recursively including children of children,
        recursively, whenever a member is a non-primitive.
     - `expanded_graph` -- Similar to the input `graph`, but whenever a
       non-primitive appears in a list of children, that non-primitive's
       children are added, and so on with children of children recursively.
    """
    # Create a pairlist and expanded graph representation of interactions.
    pairs = []
    expanded_graph = {}
    for index in graph:
        indices = recursively_expand(index, graph, expanded_graph)
        new_interactions = [
            d2v.d2v_id.unordered_pair(id1, id2) 
            for id1, id2 in it.combinations(indices, 2)
        ]
        pairs.extend(new_interactions)

    return pairs, expanded_graph


def recursively_expand(obj_id, object_graph, expanded):

    # Check the cache to see if we already expanded it.
    if obj_id in expanded:
        return expanded[obj_id]

    # Otherwise expand it!
    expanded[obj_id] = [obj_id]
    for child_id in object_graph[obj_id]:
        if child_id in object_graph:
            expanded[obj_id].extend(
                recursively_expand(child_id, object_graph, expanded)
            )
        else:
            expanded[obj_id].append(child_id)

    return expanded[obj_id]


if __name__ == "__main__":
    pass
    # read dataset.
    # initialize random embeddings
    # stream the objects
    # Cause updates as you stream



