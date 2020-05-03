import os
from collections import Counter, defaultdict
from unittest import TestCase, main
import scipy.sparse
import numpy as np
import d2v
import shutil
import itertools as it
import json


class TestD2vId(TestCase):

    def test_validate_id(self):
        # IDs must be strings.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id(5)

        # Two commas separate the id into type, field, name.
        # It's okay for field to be empty.
        d2v.d2v_id.validate_id('type,,name')
        d2v.d2v_id.validate_id('type,fieldname,name')

        # type and name can't be empty
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id(',field,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,field,')

        # There must be exactly two commas.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,field,name,comma')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,name')

        # No tabs allowed anywhere.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type\t,field,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,field\t,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,field,name\t')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type\t,,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,,name\t')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,\t,name')


    def test_validate_primitive_id(self):
        """
        This validate_primitive_id should raise errors in almost the same way
        as validate_id, except that it does not allow fields to be blank. 
        """
        # IDs must be strings.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id(5)

        # Two commas separate the id into type, field, name.
        d2v.d2v_id.validate_primitive_id('type,field,name')

        # type and name can't be empty
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id(',field,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,field,')

        # Field also cannot be empty for primitives.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,,name')

        # There must be exactly two commas.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,field,name,comma')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,name')

        # No tabs allowed anywhere.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type\t,field,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,field\t,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,field,name\t')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type\t,,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,,name\t')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_primitive_id('type,\t,name')


    def test_validate_non_primitive_id(self):
        """
        This validate_primitive_id should raise errors in almost the same way
        as validate_id, except that it does not allow fields to be blank. 
        """
        # IDs must be strings.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id(5)

        # Two commas separate the id into type, field, name.  For non-primitive
        # IDs, field must be blank.
        d2v.d2v_id.validate_non_primitive_id('type,,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type,field,name')

        # type and name can't be empty
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id(',,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_id('type,,')

        # There must be exactly two commas.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type,field,name,comma')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type,name')

        # No tabs allowed anywhere.
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type\t,,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type,\t,name')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_non_primitive_id('type,,name\t')


    def test_validate_type(self):
        """
        Object types should be non-empty strings with no commas or tabs
        """
        d2v.d2v_id.validate_type('type')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type(5)
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type('type\t')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type('type,')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type('')


    def test_validate_field(self):
        """
        Field names be non-empty strings with no commas or tabs.
        """
        d2v.d2v_id.validate_type('field')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type(5)
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type('field\t')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type('field,')
        with self.assertRaises(ValueError):
            d2v.d2v_id.validate_type('')


    def test_is_primitive(self):
        self.assertTrue(d2v.d2v_id.is_primitive('profile,title,aws'))
        self.assertFalse(d2v.d2v_id.is_primitive('skill,,1'))


    def test_unordered_pair(self):
        """
        unordered_pair normalizes the expression of cooccurrences (pairs), by
        always representing them with the two elements in sorted (ascending)
        order.
        """
        pair = d2v.d2v_id.unordered_pair(1, 2)
        expected_pair = (1, 2)
        self.assertEqual(pair, expected_pair)

        pair = d2v.d2v_id.unordered_pair(2, 1)
        expected_pair = (1, 2)
        self.assertEqual(pair, expected_pair)

        pair = d2v.d2v_id.unordered_pair('b', 'a')
        expected_pair = ('a', 'b')
        self.assertEqual(pair, expected_pair)


    def test_as_list(self):
        """
        `as_list` should convert all values into lists.
         - lists should be passed through unchanged.
         - strings should be lowercased and tokenized into lists.
         - all other objects should be wrapped in a list.
        """
        val = 'One Two'
        as_list = d2v.d2v_id.as_list(val)
        expected_as_list = ['one', 'two']
        self.assertEqual(as_list, expected_as_list)

        val = ['One', 'Two']
        as_list = d2v.d2v_id.as_list(val)
        expected_as_list = ['one', 'two']
        self.assertEqual(as_list, expected_as_list)

        val = 1
        as_list = d2v.d2v_id.as_list(val)
        expected_as_list = [1]
        self.assertEqual(as_list, expected_as_list)

        val = [1, 'Two']
        as_list = d2v.d2v_id.as_list(val)
        expected_as_list = [1, 'two']
        self.assertEqual(as_list, expected_as_list)

        val = {'$ref': 'profile,,1'}
        as_list = d2v.d2v_id.as_list(val)
        expected_as_list = [{'$ref': 'profile,,1'}]
        self.assertEqual(as_list, expected_as_list)


class TestGraph(TestCase):

    def test_read_write_object_graph(self):
        object_graph1 = {
            'joblist,,1': ['profile,,1'],
            'profile,,1': [
                'profile,title,aws', 'profile,title,engineer',
                'profile,title,for', 'profile,title,ai',
                'profile,title,research', 'skill,,1', 'skill,,2'
            ], 
            'skill,,1': ['skill,name,orchestration', 'skill,category,ops'],
            'skill,,2': ['skill,name,jenkins', 'skill,category,devops']
        }
        path = os.path.join(
            d2v.CONSTANTS.TEST_DIR, 'test-read-write-graph.tsv'
        )
        if os.path.exists(path):
            os.remove(path)
        d2v.graph.write_object_graph(path, object_graph1)
        object_graph2 = d2v.graph.read_object_graph(path)
        self.assertEqual(object_graph2, object_graph1)
        os.remove(path)


    def test_index_graph(self):

        graph = TestData.graph()
        prim_dict = TestData.prim_dict()
        non_prim_dict = TestData.non_prim_dict()
        indexed_graph = d2v.graph.index_graph(graph, prim_dict, non_prim_dict)

        # First, ensure that both graphs have the same number of items.
        self.assertEqual(len(indexed_graph), len(graph))

        # Now, check that each item in the graph is as expected.
        for parent_id in graph:

            # Convert parent's ID to index.
            parent_index = len(prim_dict) + non_prim_dict[parent_id]

            # Convert the child IDs to indices.
            expected_child_indices = [
                prim_dict[child_id] 
                if d2v.d2v_id.is_primitive(child_id) else
                len(prim_dict) + non_prim_dict[child_id]
                for child_id in graph[parent_id]
            ]

            # Check that the indexed graph has the child_indices expected for
            # the given parent index.
            child_indices = indexed_graph[parent_index]
            self.assertEqual(child_indices, expected_child_indices)


    def test_graph_to_csr(self):

        # In our test Data, `graph` will only have binary values.  So use
        # the `expanded_graph`, which has some values greater than 1 for a 
        # more complete test.
        graph = TestData.graph()
        pairs, expanded_graph = d2v.ingestion.make_pairs_and_expanded_graph(
            graph
        )

        prim_dict = TestData.prim_dict()
        non_prim_dict = TestData.non_prim_dict()
        size = len(prim_dict) + len(non_prim_dict)
        shape = (size, size)
        dtype = 'float32'

        # Assume that d2v.graph.index_graph works.
        indexed_graph = d2v.graph.index_graph(
            expanded_graph, prim_dict, non_prim_dict
        )

        # Perform the function to be tested.
        adjacency_csr = d2v.graph.graph_to_csr(
            indexed_graph, shape=shape, dtype=dtype
        )

        # Test the shape.
        expected_shape = (size, size)
        self.assertEqual(adjacency_csr.shape, expected_shape)

        # Test the dtype.
        self.assertEqual(str(adjacency_csr.dtype), dtype)

        # Test the contents of the adjacency matrix
        # Compare each row of adjacency back to the indexed_graph.
        # Convert the adjacency row into a counter dict for comparison. 
        for row in range(size):

            # Skip empty rows of csr matrix.
            row_data = adjacency_csr[row].data
            if len(row_data) == 0:
                continue

            # Convert the row into a Counter to compare to `indexed_graph`.
            _, cols = adjacency_csr[row].nonzero()
            row_as_counts = {
                col : datum
                for col, datum in zip(cols, row_data)
            }

            # Test that contents are correct.
            expected_row_as_counts = Counter(indexed_graph[row])
            self.assertEqual(row_as_counts, expected_row_as_counts)



class TestIngest(TestCase):


    def test_ingest(self):

        # Clear the path, and run the function being tested.
        path = os.path.join(d2v.CONSTANTS.TEST_DIR, 'test-ingest')
        ensure_dir(path)
        d2v.ingestion.ingest(TestData.object_iterable(), path)

        # Check that graph was correctly encoded as an adjacency matrix
        graph_adjacency_path = os.path.join(path, 'graph.npz')
        graph_adjacency = scipy.sparse.load_npz(graph_adjacency_path)
        self.assertTrue(np.allclose(
            graph_adjacency.todense(), 
            TestData.graph_adjacency().todense()
        ))

        # Ensure primitive dictionary was properly calculated and written.
        dict_path = os.path.join(path, 'dictionary.txt')
        found_dictionary = d2v.dictionary.read_dictionary(dict_path)
        #with open(dict_path) as dict_file:
        #    found_dictionary = {
        #        line.strip() : i for i, line in enumerate(dict_file)
        #    }
        self.assertEqual(found_dictionary, TestData.dictionary())

        # Check that pairs were recorded correctly as list.
        # (Equality is only up to number of occurrences, not ordering.)
        pair_path = os.path.join(path, 'pairs.tsv')
        pair_list = list(d2v.pairlist.read_pairlist(pair_path))
        self.assertEqual(Counter(pair_list), Counter(TestData.pair_list()))

        # Check that pairs were recorded correctly as a sparse matrix
        pair_adjacency_path = os.path.join(path, 'pairs.npz')
        pair_adjacency = scipy.sparse.load_npz(pair_adjacency_path)
        self.assertTrue(np.allclose(
            pair_adjacency.todense(), 
            TestData.pair_adjacency().todense()
        ))

        # Check that expanded graph adjacency was correctly encoded and written.
        expanded_graph_adjacency_path = os.path.join(path, 'expanded-graph.npz')
        expanded_graph_adjacency = scipy.sparse.load_npz(
            expanded_graph_adjacency_path
        )
        expected = TestData.expanded_graph_adjacency()
        self.assertTrue(np.allclose(
            expanded_graph_adjacency.todense(), 
            TestData.expanded_graph_adjacency().todense()
        ))


    def test_make_pairs_and_expanded_graph(self):
        """
        Test that pairs and expanded_graph are calculated correctly.  Assumes
        that recursively_expand calculates correctly.

        `pairs` should be calculated as follows
         - Every pair combination that can be made between children that are
            both contained in the same object should be listed somewhere in 
            the pairlist.
         - "contained" above means either as a direct child, or a descendent.
         - If a child occurs inside an object multiple times, then it forms
            pairs multiple times (multiple instances of the same pair must be
            logged in pairs), and that child can interact with itself.

         `expanded_graph` should be calculated as follows
         - It is a dictionary whose keys are d2v_ids for non-primitive objects
         - the values associated to each key is a list of the d2v_ids of the
            non-primitive object's descendents.
         - An object A is a descendent of an object B if A is the direct child
           of B or is a child of a child, recursively.
         - If A is a descendent fo B in multiple ways, it should appear an
            equal number of times in the list of descendents of B.
        """

        # Run the target function on our test data.
        graph = TestData.graph()
        pairs, expanded_graph = d2v.ingestion.make_pairs_and_expanded_graph(
            graph
        )

        # Assemble what we expect to have gotten, assuming that recursively
        # expand works (it is covered by a unit test).
        expected_expanded_graph = {}
        expected_pairs = []
        for parent_id in graph:
            descendents = d2v.ingestion.recursively_expand(
                parent_id, graph, expected_expanded_graph)
            expected_pairs.extend(
                d2v.d2v_id.unordered_pair(child_id1, child_id2)
                for child_id1, child_id2 in it.combinations(descendents, 2)
            )

        self.assertEqual(Counter(pairs), Counter(expected_pairs))
        self.assertEqual(expanded_graph, expected_expanded_graph)



    def test_recursively_expand(self):

        d2v_id = 'joblist,,1'
        d2v_ids = d2v.ingestion.recursively_expand(d2v_id, TestData.graph(), {})
        expected_d2v_ids = [
            'joblist,,1', 
            'profile,,1',
            'profile,title,aws', 'profile,title,engineer',
            'profile,title,for', 'profile,title,ai',
            'profile,title,research', 
            'skill,,1', 'skill,name,orchestration', 'skill,category,ops',
            'skill,,2', 'skill,name,jenkins', 'skill,category,devops',
            'profile,,2', 'profile,title,software',
            'profile,title,developer', 'profile,title,for',
            'profile,title,deep', 'profile,title,learning',
            'profile,title,tools',
            'skill,,2', 'skill,name,jenkins', 'skill,category,devops',
            'skill,,3', 'skill,name,agile', 'skill,category,dev',
        ]
        # Test that the contents are the same up to reordering.
        self.assertEqual(Counter(d2v_ids), Counter(expected_d2v_ids))

        d2v_id = 'profile,,1'
        d2v_ids = d2v.ingestion.recursively_expand(d2v_id, TestData.graph(), {})

        expected_d2v_ids = [   
            'profile,,1',
            'profile,title,aws', 'profile,title,engineer',
            'profile,title,for', 'profile,title,ai',
            'profile,title,research', 
            'skill,,1', 'skill,name,orchestration', 'skill,category,ops',
            'skill,,2', 'skill,name,jenkins', 'skill,category,devops'
        ]
        self.assertEqual(Counter(d2v_ids), Counter(expected_d2v_ids))

        d2v_id = 'skill,,1'
        d2v_ids = d2v.ingestion.recursively_expand(d2v_id, TestData.graph(), {})
        expected_d2v_ids = [
            'skill,,1', 'skill,name,orchestration', 'skill,category,ops',
        ]
        self.assertEqual(Counter(d2v_ids), Counter(expected_d2v_ids))

        # Cannot recursively expand a child.
        with self.assertRaises(KeyError):
            d2v_id = 'skill,name,ops'
            d2v_ids = d2v.ingestion.recursively_expand(
                d2v_id, TestData.graph(), {})
            expected_d2v_ids = ['skill,name,ops']
            self.assertEqual(Counter(d2v_ids), Counter(expected_d2v_ids))



class TestPairlist(TestCase):

    def test_pairlist_to_coo(self):

        pairs = Counter({
            (0, 7): 3, (1, 7): 2, (2, 7): 1, (3, 7): 1, (4, 7): 1, (5, 7): 1,
            (6, 7): 1, (5, 8): 1, (5, 9): 1, (6, 10): 1, (6, 11): 1
        })
        symmetric_shape = (12,12)

        coo_matrix = d2v.pairlist.pairlist_to_coo(
            pairs, shape=symmetric_shape, symmetric=True
        )

        # Check the symmetry condition.
        self.assertTrue(np.allclose(
            coo_matrix.todense(), coo_matrix.T.todense()))

        # Check correctness of the values.
        recovered_pairs = {
            d2v.d2v_id.unordered_pair(i, j): int(d)
            for i,j,d in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
        }
        self.assertEqual(pairs, recovered_pairs)


    def test_read_write_pairlist(self):
        path = os.path.join(
            d2v.CONSTANTS.TEST_DIR,
            'test-read-write-pairslist.csv'
        )
        pair_list = TestData.pair_list()
        d2v.pairlist.write_pairlist(path, pair_list)
        read_pair_list = list(d2v.pairlist.read_pairlist(path))
        self.assertEqual(read_pair_list, pair_list)



class TestDictionary(TestCase):

    def test_dictionary(self):
        key1 = 'a'
        key2 = 'b'
        key3 = 'c'
        key4 = 'd'
        dictionary = d2v.dictionary.Dictionary()

        int_id = dictionary.add(key1)
        self.assertEqual(int_id, 0)

        int_id = dictionary.add(key2)
        self.assertEqual(int_id, 1)

        int_id = dictionary.add(key1)
        self.assertEqual(int_id, 0)

        int_ids = dictionary.add_many([key3, key4, key1, key2])
        expected_int_ids = [2, 3, 0, 1]
        self.assertEqual(int_ids, expected_int_ids)

        keys = [dictionary.keys[int_id] for int_id in range(4)]
        expected_keys = [key1, key2, key3, key4]
        self.assertEqual(keys, expected_keys)


    def test_read_write_dictionary(self):
        keys = ['a', 'b', 'c', 'd']
        path = os.path.join(d2v.CONSTANTS.TEST_DIR, 'read-write-dictionary.txt')
        dictionary1 = d2v.dictionary.Dictionary()
        dictionary1.add_many(keys)

        d2v.dictionary.write_dictionary(path, dictionary1)
        with open(path) as dictionary_file:
            dictionary_text = dictionary_file.read()
        expected_dictionary_text = '\n'.join(keys) + '\n'

        dictionary2 = d2v.dictionary.read_dictionary(path)
        self.assertEqual(dictionary2.keys, keys)



class TestData:

    """Access a small, consistent test dataset."""

    path = 'test-data/simple-data'

    @classmethod
    def object_iterable(cls):
        path = os.path.join(cls.path, 'object-iterator.json')
        with open(path) as f:
            return json.loads(f.read())

    @classmethod
    def dictionary(cls):
        path = os.path.join(cls.path, 'dictionary.txt')
        return d2v.dictionary.read_dictionary(path)

    @classmethod
    def graph_adjacency(cls):
        path = os.path.join(cls.path, 'graph.npz')
        return scipy.sparse.load_npz(path)

    @classmethod
    def pair_list(cls):
        path = os.path.join(cls.path, 'pairs.tsv')
        return d2v.pairlist.read_pairlist(path)

    @classmethod
    def expanded_graph_adjacency(cls):
        path = os.path.join(cls.path, 'expanded-graph.npz')
        return scipy.sparse.load_npz(path)

    @classmethod
    def pair_adjacency(cls):
        path = os.path.join(cls.path, 'pairs.npz')
        return scipy.sparse.load_npz(path)


 


def ensure_dir(path):
    """
    Make sure that path is an empty directory.  
    Remove anything that was already there.
    """
    clear_path(path)
    os.makedirs(path)


def clear_path(path):
    "Remove anything at path."
    if os.path.exists(path):
        shutil.rmtree(path)


if __name__ == '__main__':
    main()
