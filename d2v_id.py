import d2v
import re

# Must have at least one character and no commas.
VALID_TYPE = re.compile('[^,\t]+$')

# Must have at least one character and no commas.
VALID_FIELD = re.compile('[^,\t]+$')

# Must have at least one character.
VALID_NAME = re.compile('[^,\t]+$')

# First two commas split the string into three parts:
#   1) A valid type
#   2) A valid field, or empty string (no field for non-primitives)
#   3) A valid name.
VALID_ID = re.compile('[^,\t]+,([^,\t]+)?,[^,\t]+$')

# First two commas split the string into three parts:
#   1) A valid type
#   2) A valid field (no empty string, all primitives have a field).
#   3) A valid name.
VALID_PRIMITIVE_ID = re.compile('[^,\t]+,[^,\t]+,[^,\t]+$')

# First two commas split the string into three parts:
#   1) A valid type
#   2) An empty string (no field for non-primitives)
#   3) A valid name.
VALID_NON_PRIMITIVE_ID = re.compile('[^,\t]+,,[^,\t]+$')


def validate_field(field):
    """Fields can be either None, or a conformant string."""
    if field is not None:
        validate(field, VALID_FIELD)


def validate_type(d2v_type):
    validate(d2v_type, VALID_TYPE)


def validate_id(d2v_id):
    validate(d2v_id, VALID_ID)


def validate_primitive_id(d2v_id):
    validate(d2v_id, VALID_PRIMITIVE_ID)


def validate_non_primitive_id(d2v_id):
    validate(d2v_id, VALID_NON_PRIMITIVE_ID)


def validate(string, regex):
    if not isinstance(string, str) or not regex.match(string):
        raise ValueError(
            'Invalid format.  expecting "{}".  Got "{}" ({}).'
            .format(regex.pattern, string, type(string).__name__)
        )


def split_id(d2v_id):
    obj_type, field, name = d2v_id.split(',', 2)
    if field == '':
        field = None
    return obj_type, field, name


def is_primitive(d2v_id):
    """ 
    Return True if the d2v_id corresponds to a primitive, False otherwise.
    """
    obj_type, field, name = split_id(d2v_id)
    return field is not None


def join_id(obj_type, field, obj):
    """
    Construct the d2v_id from an obj_type, field name, and the object itself.
    """
    # Handle the case where `obj` is non-primitive, or is a 
    # reference to a non-primitive.
    if isinstance(obj, dict):
        return get_non_primitive_id(obj)

    # Otherwise the ID is formed by joining the context and string
    # representation of the object.  First validate the context.
    validate_type(obj_type)
    validate_field(field)

    # Serialize and join.
    if field is None:
        field = ''
    name = str(obj)
    d2v_id = ','.join((obj_type, field, name))

    return d2v_id


def get_non_primitive_id(obj):
    """
    Obtain a d2v_id from a non-primitive object.
    """
    if 'd2v-id' in obj:
        return obj['d2v-id']
    elif '$ref' in obj:
        return obj['$ref']
    else:
        raise ValueError(
            "Cannot get d2v-id, improperly formatted non-primitive object"
        )


def unordered_pair(id1, id2):
    """Return a tuple with the smaller ID first."""
    if id1 < id2:
        return id1, id2
    return id2, id1


def get_child_ids(obj):

    parent_id = d2v.d2v_id.get_non_primitive_id(obj)
    obj_type, field, name = d2v.d2v_id.split_id(parent_id)
    d2v.d2v_id.validate_type(obj_type)
    child_ids = []
    for field, expression in obj.items():

        # Skip d2v metadata fields.
        if field == 'd2v-id': 
            continue

        # All fields are handled treated as lists of values.  
        value_list = as_list(expression)

        # Now get the id for each embeddable object in value_list.
        child_ids.extend([
            d2v.d2v_id.join_id(obj_type, field, val) for val in value_list
        ])

    return child_ids


def as_list(expression):
    if isinstance(expression, list):
        return [normalize(element) for element in expression]
    elif isinstance(expression, str):
        return [normalize(token) for token in tokenize(expression)]
    else:
        return [normalize(expression)]


def normalize(element):
    """
    Normalize the representation of values.  
    This lowercases strings and leaves all other elements unchanged.
    """
    if isinstance(element, str):
        return element.lower()
    return element


def tokenize(string):
    return string.split()


