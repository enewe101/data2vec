"""
Given the set of entities found in the primitive and non-primitive dictionaries, - allocate learned parameters for each entity
 - provide access to subets of the vectors corresponding to the entities
    of a given type-field.
 - provide head-tail encoding for large vocabularies
"""
import numpy as np
from collections import defaultdict



