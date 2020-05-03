import d2v
import scipy.sparse


#def read_object_graph(path):
#    object_graph = {}
#    with open(path) as object_file:
#        for line in object_file:
#            all_d2v_ids = line.strip().split('\t')
#            parent_id, child_ids = all_d2v_ids[0], all_d2v_ids[1:]
#            object_graph[parent_id] = child_ids
#    return object_graph
#
#
#def write_object_graph(path, object_graph):
#    with open(path, 'w') as object_file:
#        for parent_id, child_ids in object_graph.items():
#            object_file.write(parent_id + '\t' + '\t'.join(child_ids) + '\n')


#def index_graph(graph, prim_dict, non_prim_dict):
#    """
#    Create a copy of `graph` that uses indexes provided by the
#    dictionaries rather than d2v_ids.
#
#    Inputs
#     - graph - dict<list<str>> - a dictionary in which keys are parent IDs
#        and values are the list of IDs belonging to children.
#     - prim_dict - d2v.dictionary.Dictionary that assigns integer indexes to
#        primitive objects.
#     - non_prim_dict - d2v.dictionary.Dictionary that assigns integer
#        indexes to non-primitive objects.
#    """
#    new_graph = {}
#    for parent_id, child_ids in graph.items():
#        parent_index = get_index(parent_id, prim_dict, non_prim_dict)
#        child_indices = [
#            get_index(cid, prim_dict, non_prim_dict) 
#            for cid in child_ids
#        ]
#        new_graph[parent_index] = child_indices
#    return new_graph


def graph_to_csr(graph, shape=None, dtype=int):
    """
    Convert the `graph` into an adjacency matrix.
    """
    data, I, J = [], [], []
    for parent_index, child_indices in graph.items():
        data.extend([1 for _ in range(len(child_indices))])
        I.extend([parent_index for _ in range(len(child_indices))])
        J.extend(child_indices)
    return scipy.sparse.csr_matrix(
        (data, (I, J)), shape=shape, dtype=dtype
    )


def get_index(the_id, prim_dict, non_prim_dict):
    if d2v.d2v_id.is_primitive(the_id):
        return prim_dict[the_id]
    return non_prim_dict[the_id] + len(prim_dict)

