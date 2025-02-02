import os
import numpy as np

DATABASE_FILE = "face_database.npz"

def save_embedding(name, embedding):
    """
    Saves the given embedding for the user 'name' in a local NPZ file.
    """
    # Check if the database file exists
    if os.path.exists(DATABASE_FILE):
        data = np.load(DATABASE_FILE, allow_pickle=True)
        names = data["names"].tolist()
        embeddings = data["embeddings"].tolist()
    else:
        names = []
        embeddings = []

    # Append new data
    names.append(name)
    embeddings.append(embedding)

    # Save updated data
    np.savez(DATABASE_FILE, names=names, embeddings=embeddings)

def load_database():
    """
    Loads the embeddings database and returns two lists: names and embeddings.
    """
    if not os.path.exists(DATABASE_FILE):
        return [], []
    
    data = np.load(DATABASE_FILE, allow_pickle=True)
    names = data["names"].tolist()
    embeddings = data["embeddings"].tolist()
    return names, embeddings
