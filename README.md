
Data representation in preparation for embedding.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We want to allow just about any data structure that is built up from the 
usual primitives (ints, floats, strings) and collections (lists and dicts).

JSON, and many other formats are up to the task.  We allow referencing other
objects, defined elsewhere, using JSON references and user-provided ids.
Here is an example of an object containing a reference:

{
	'j2v_id': 'profile,,1'
	'name':	'Jorg Neussen',
	'address': {'$ref': 'address,,1'}
}

Here, the address field refers to some other object, which may be defined 
elsewhere.

To create embeddings, we need to boil all data structures down into some set of
pairwise interaction strengths between primitives.  Probably the most natural
way to represent this is as a dictionary having pairs of object IDs as keys,
and ints or floats as values representing interaction strengths---or in any
other sparse-matrix-like representation.


Tokenizing
~~~~~~~~~~

In the example above, We have a primitive stored in the 'name' field.  j2v
tokenizes by default, so one of the first steps to ingesting this object is to
tokenize

{
	'j2v_id': 'profile,,1'
	'name':	['jorg', 'neussen'],	# <-- got tokenized / normalized
	'address': {'$ref': 'address,,1'}
}


Notions of interaction
~~~~~~~~~~~~~~~~~~~~~~

There are many notions of interaction one could apply.  In this case, we
consider two objects to interact if they occur as part of the same collection.
So one interaction within our example is between 'jorg' and 'neussen', because
they are part of the same list.

Representing interactions
~~~~~~~~~~~~~~~~~~~~~~~~~

There are also other interactions in the example above, but to discuss them
effectively, we need a way to denote interactions.  We do so using pairs of IDs.
IDs in j2v are namespaced in a specific way.  To denote the interaction between
'Jorg' and 'Neussen', we write:

	( 'profile,name,jorg' , 'profile,name,neussen' )

The prefixes remind us of the context in which the data was found.  This 
preserves the possibility of distinguishing between, say, the word "quarantine" 
occurring in the field 'current\_status' vs. in the field 'desired\_status'.

This also facilities building a contextualized vocabulary and entity frequency
table, both of which are helpful for sample-based algorithms.  Sample based
algorithms generally need to sample from the vocabulary, either based on 
unigram frequency or current model distribution.  IDs with the structure
'type,field,value' let us do that.


Graph representation
~~~~~~~~~~~~~~~~~~~~
This way of converting input data into pairs is destructive.  It would be 
difficult to recover the the original data from the pairs alone.  One thing
that is difficult to tell at a glance is the hierarchical structure---what
contained what.  So we may know that A was together with B, but we don't know
what contained them.  We may be able to infer it, but it would be difficult.

To retain this information, we cache the input data into a graph representation.
As it is accumulating, we use a dict in which keys represent source nodes and
values contain a list of target nodes.  Sources represent non-primitive objecs 
(lists and dicts) and the targets represent the objects that are members of 
the non-primitives.  It is also planned that this will allow encoding some 
other structure, e.g. that two lists of values represent parallel arrays,
listing properties for a sequence of objects (e.g. a list of tokens and a
parallel list of POS tags).  The purpose of the graph data structure is to
capture as much of the original data structure as is relevant for j2v without
applying any specific notion of cooccurrence counting.

To see the graph representation at work, we need to add more data to our 
example.  Take these two objects as inputs:

{
	'j2v_id': 'profile,,1'
	'name':	'Jorg Neussen',
	'address': {'$ref': 'address,,1'}
}

{
	'j2v_id': 'address,,1',
	'street_name':	'St. Laurent Blvd.',
	'number': 1337,
	'postal_code': 'Y0Y 0Y0'
}

This generates the following object graph:
{
	'profile,,1': ['profile,name,jorg', 'profile,name,neussen', 'address,,1'],
	'address,,1': [
		'address,street_name,st.', 'address,street_name,laurent',
		'address,street_name,blvd.', 'address,number,1337', 
		'address,postal_code,y0y', 'address,postal_code,0y0'
	]
}

There are a couple things to point out here.  First, 'address,,1' did not
get namespaced under 'profile,address'.  Referenced objects create their own
namespace.

The other thing to notice is that we may want to treat the 'number' field in
the address differently, or not split the postal code.  Any schema-specific
considerations like that should be applied before obtaining the object
graph.  Thus the object graph provides j2v view on the data, but does not
commit to a specific notion of cooccurrence nor cater to a specific learner.

While the dict representation of the graph is useful, the kinds of operations
that learners need to apply are usually better served by expressing the graph
as a sparse adjacency matrix.  

To store the graph's adjacency matrix, we need to assign each primitive and 
non-primitive object to some row (and column) of the matrix.  Then, we 
write a 1 at (i,j) if the object corresponding to row i contains the object
corresponding to column j, and 0 otherwise.

This assignment is taken care of by means of two dictionaries, the primitive
dictionary and non-primitive dictionary (or prim\_dict and non\_prim\_dict).
These two dictionaries assign a contiguous set of integers (array positions) 
starting from zero to (separately) primitives and non-primitives separately.
keeping the dictionaries separate makes it easy to establish an ordering
in which all primitives come before all non-primitives, which is handy for
some learners and calculating some notions of cooccurrence.  It allows portions
of the matrix corresponding to primitives (or non-primitives) to be easily
sliced out of the matrix.

Whenever we calculate the cooccurrence frequency (or pairs), we can also store
it in a sparse matrix using the same assignment of objects to ids.  We'll
generally refer to this as the cooccurrence matrix.

One final representation of the input data called the descendency graph is
calculated because it helps some learners.  This is a graph in which the 
value stored in (i,j) indicates whether j is a descendent of i.
An object, j, is a descendent of another object, i, if j is a child of i, or a
child of a child of i, etc.  It is possible for j to arise multiple times as a
descendent of i (multiple copies might occur in i, or more than one child of i
may reference objects containing j.  So, the value stored in (i,j) is the
number of times that j appears among the descendents of i.

Rating data
A rating (e.g. a user rating a movie), provides an interaction between a rating
object and a user object.  Unlike for other interactions, where mere presence
together is counted, here we look to the score given by the user in the rating
as a strength of association between that user and what they rated.  The
ability to interpret one field as the strength of association between entities
defined in other fields will be important.


A few kinds of algorithm to look at:
1) SVD.  Works on the 
2) ALS.  Alternating least squares. Used in collaborative filtering. 
	requires no more
3) GBT.  Gilbert.  Hilbert with Gibbs Sampling.

Several characteristics to consider:
1) Accuracy
2) Training time / cost
3) Stability
4) On-line learning ability

Notions of Cooccurrence
1) Only primitives get learned vector representations, the vector
	representation for non-primitives are defined to be the sum of those of
	their children.  Children of the same object cooccur, and cooccurrence
	between non-primitives is treated like a cooccurrence among all of their
	children.

2) Non-primitives also get learned vector representations; children interact
	with siblings as above, but (perhaps) children also interact with their
	parent's learned representation.  When a non-primitive interacts, the
	interaction is taken to be with the non-primitives own embedding plus the
	sum of its children's embeddings.

3) Probably many are possible, these seem like a good start.  May make no
	difference.

Gilbert learner variation.
The object graph could be exploited to calculate gilbert updates more 
quickly.  To apply the update implied by the interactions between all members
of an object, first tally all members' vectors, then interact each member with
this total vector once.  This reduces n^2/2 - n/2 operations to 2n.


