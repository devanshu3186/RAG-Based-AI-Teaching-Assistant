from sentence_transformers import util
import numpy as np
import joblib
from read_chunks import create_embeddings

# Loading the saved DataFrame
df=joblib.load("chunks_embedding.pkl")

incoming_query=input("Ask a Question: ")
question_embedding=create_embeddings(incoming_query)

# Find similarities of question embedding with other embeddings 

# Combine all separate embedding arrays into one 2D array of shape (number_of_chunks, 768)
chunk_embeddings_array = np.stack(df["embedding"].values) 

similarities = util.cos_sim(question_embedding, chunk_embeddings_array).flatten()
print(similarities)
top_results=4
max_idx=similarities.argsort(descending=True)[:top_results].tolist()
print(max_idx)
new_df=df.loc[max_idx]
print(new_df[["title", "number", "text"]])