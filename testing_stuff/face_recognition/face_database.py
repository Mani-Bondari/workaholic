import os
import numpy as np

DATABASE_FILE = "face_database.npz"

def save_embedding(name, embedding):
    """
    Saves the given embedding (1D numpy array) for the user 'name'
    in a local NPZ file. If the person already exists, multiple embeddings
    are stored as a list.
    """
    # Load existing database if it exists
    if os.path.exists(DATABASE_FILE):
        # Important to use allow_pickle=True because we're storing lists of arrays.
        data = np.load(DATABASE_FILE, allow_pickle=True)
        # Reconstruct the dictionary { name -> [embedding1, embedding2, ...] }
        names = data["names"]
        emb_lists = data["embeddings"]  # This is an object array, each element is a list of embeddings

        embeddings_dict = {}
        for i, n in enumerate(names):
            embeddings_dict[n] = list(emb_lists[i])
    else:
        embeddings_dict = {}

    # Append the new embedding to the appropriate list
    if name not in embeddings_dict:
        embeddings_dict[name] = []

    embeddings_dict[name].append(np.array(embedding, dtype=np.float32))

    # Convert back to arrays for saving
    all_names = list(embeddings_dict.keys())
    all_embeds = list(embeddings_dict.values())

    # We store each value in all_embeds as a list, which we convert to object arrays
    np.savez_compressed(
        DATABASE_FILE,
        names=np.array(all_names, dtype=object),
        embeddings=np.array([np.array(lst, dtype=object) for lst in all_embeds], dtype=object),
    )

def load_database():
    """
    Loads the embeddings database and returns a dictionary:
      { name: [embedding1, embedding2, ...], ... }
    Each embedding is a numpy array of shape (512,).
    """
    if not os.path.exists(DATABASE_FILE):
        return {}

    data = np.load(DATABASE_FILE, allow_pickle=True)
    names = data["names"]
    emb_lists = data["embeddings"]

    embeddings_dict = {}
    for i, n in enumerate(names):
        # Convert each stored list of embeddings to float32 numpy arrays
        stored_list = emb_lists[i]  # This itself is a list (or object array) of embeddings
        embeddings_dict[n] = [np.array(e, dtype=np.float32) for e in stored_list]

    return embeddings_dict
