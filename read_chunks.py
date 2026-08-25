from sentence_transformers import SentenceTransformer, util
import json
import os
import pandas as pd
import numpy as np

# Load a lightweight, popular open-source model
model = SentenceTransformer("all-mpnet-base-v2")
def create_embeddings(text):

    embedding = model.encode(text)  # Generate vector embedding

    # embedding_list=embedding.tolist()  # numpy array to list conversion

    return  embedding


json_dir = sorted(os.listdir("json"))

my_dicts=[]
chunk_id=0
for json_file in json_dir:
    with open(f"json/{json_file}", "r") as f:
        data=json.load(f)
    print(f"Creating embeddings for {json_file}")
    for chunk in data["chunks"]:
        chunk["chunk_id"]=chunk_id
        chunk["embedding"]=create_embeddings(chunk["text"])
        chunk_id+=1
        my_dicts.append(chunk)

    break
    

df=pd.DataFrame.from_records(my_dicts)


incoming_query=input("Ask a Question: ")
question_embedding=create_embeddings(incoming_query)


# Find similarities of question embedding with other embeddings 

# chunk_embeddings_list = df["embedding"].values.tolist()  #Slow approach

chunk_embeddings_array = np.stack(df["embedding"].values) # Combine all separate embedding arrays into one 2D array of shape (number_of_chunks, 768)

similarities = util.cos_sim(question_embedding, chunk_embeddings_array).flatten()
print(similarities)
top_results=4
max_idx=similarities.argsort(descending=True)[:top_results]
print(max_idx)
new_df=df.loc[max_idx]
print(new_df[["title", "number", "text"]])