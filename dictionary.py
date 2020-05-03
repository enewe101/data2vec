
def read_dictionary(path):
    """
    Reads in a dictionary that has been recorded by writing one key per line,
    with each key on the line corresponding to its integer id.  No validation
    (e.g. against repeated keys) is performed.
    """
    dictionary = Dictionary()
    with open(path) as dictionary_file:
        for i, line in enumerate(dictionary_file):
            key = line.strip()
            dictionary.add(key)
    return dictionary


def write_dictionary(path, dictionary):
    """
    Writes a record of the dictionary by writing each key followed by a newline
    in order.  Each key is written on the line corresponding to its integer id.
    """
    with open(path, 'w') as dictionary_file:
        for key in dictionary.keys:
            dictionary_file.write(key + '\n')


class Dictionary(dict):
    """
    Assigns keys an autoincrementing id starting from zero.
    """
    def __init__(self):
        self.keys = []
        super().__init__()

    def add(self, key):
        if key not in self:
            self[key] = len(self)
            self.keys.append(key)
        return self[key]

    def add_many(self, key_iterable):
        return [self.add(key) for key in key_iterable]

    def get_many(self, key_iterable):
        return [self[key] for key in key_iterable]


