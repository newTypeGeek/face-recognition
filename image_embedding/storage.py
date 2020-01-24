import pickle

def deserialize(path):
    '''
    Deserialize the pickle file
    Arguments:
    1. path:  Path that stored the pickle file
    Returns:
    1. data:        Deserialized object
    '''
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def serialize(data, path):
    '''
    Serialize the data and save to disk
    Arguments:
    1. data:            Data to serialize and save
    2. output_path:     Path to save the file
    '''
    with open(path, "wb") as f:
        pickle.dump(data, f)
