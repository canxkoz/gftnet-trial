# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import get_linear_schedule_with_warmup, FNetTokenizer
from utils import init_loader

# Assuming you have a DataFrame called 'df' with a column named 'text' containing the documents

# Step 1: Preprocess the documents
# Tokenization, stop word removal, and other preprocessing steps can be done here
# Example preprocessing:


tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")
_, _, df_s = init_loader(task="sst2", max_length=8, batch_size=32)
# %%
for split in ["train", "validation", "test"]:
    df_s[split]["tokens"] = df_s[split]["input_ids"].apply(
        lambda x: tokenizer.convert_ids_to_tokens(x)
    )
    df_s[split]["tokens"] = df_s[split]["tokens"].apply(lambda x: ",".join(x))


df = df_s["validation"]
# %%

# %
# Create a document-term matrix using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","))
dtm = vectorizer.fit_transform(df["tokens"])

# Convert the document-term matrix to a dense matrix
dtm_dense = dtm.toarray()

# Compute the word co-occurrence matrix
word_cooccur = np.dot(dtm_dense.T, dtm_dense)

# Compute the word frequency in each document
doc_word_freq = np.sum(dtm_dense, axis=0)

# Compute the total word frequency in the corpus
total_word_freq = np.sum(doc_word_freq)

# Compute the PPMI matrix
ppmi_matrix = np.zeros_like(word_cooccur, dtype=np.float64)
rows, cols = ppmi_matrix.shape

for i in range(rows):
    for j in range(cols):
        pmi = np.log(
            (word_cooccur[i, j] * total_word_freq)
            / (doc_word_freq[i] * doc_word_freq[j])
        )
        ppmi = max(0, pmi)
        ppmi_matrix[i, j] = ppmi
# %%
word_to_index = vectorizer.vocabulary_

# Print the word-to-index lookup table
for word, index in word_to_index.items():
    print(f"{word}: {index}")
