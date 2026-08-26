from sentence_transformers import SentenceTransformer
import json
import os
import pandas as pd
import joblib

# Load a lightweight, popular open-source model
model = SentenceTransformer("all-mpnet-base-v2")
def create_embeddings(all_texts):

    #Process everything in batches
    embedding = model.encode(
    sentences=all_texts,
    batch_size=64,           # Adjust based on your GPU/CPU memory
    show_progress_bar=True,  # Displays a progress bar
    convert_to_numpy=True,   # Returns a numpy array
    )
    return  embedding

if __name__=="__main__":

    json_dir = sorted(os.listdir("json"))

    my_dicts=[]
    chunk_id=0
    for json_file in json_dir:
        with open(f"json/{json_file}", "r") as f:
            data=json.load(f)
        print(f"Creating embeddings for {json_file}")

        #Get the list of text from all chunks
        all_texts=[chunk["text"] for chunk in data["chunks"]]
        embeddings=create_embeddings(all_texts) 

        for i, chunk in enumerate(data["chunks"]):
            chunk["chunk_id"]=chunk_id
            chunk["embedding"]=embeddings[i]
            chunk_id+=1
            my_dicts.append(chunk)


    # Saving the DataFrame
    df=pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, "chunks_embedding.pkl")