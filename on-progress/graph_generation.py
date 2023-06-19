# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
import torch

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

cosine_similarity_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)

word_graph = nx.Graph(cosine_similarity_matrix)

adj_matrix = nx.to_numpy_array(word_graph)
tensor = torch.from_numpy(adj_matrix)
tensor = tensor.to(dtype=torch.complex64)

# %%
