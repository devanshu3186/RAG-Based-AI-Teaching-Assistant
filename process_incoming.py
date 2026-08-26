from sentence_transformers import util
import numpy as np
import joblib
from preprocess_json import create_embeddings
from ollama import chat

def inference(prompt):
    stream = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': prompt}],
    stream=True,
    think=False
    )

    return stream


# Loading the saved DataFrame
df=joblib.load("chunks_embedding.pkl")

incoming_query=input("Ask a Question: ")
question_embedding=create_embeddings(incoming_query)

# Find similarities of question embedding with other embeddings 
chunk_embeddings_array = np.stack(df["embedding"].values) 
similarities = util.cos_sim(question_embedding, chunk_embeddings_array).flatten()

top_results=10
max_idx=similarities.argsort(descending=True)[:top_results].tolist()

new_df= df.loc[max_idx]
new_df= new_df.drop_duplicates(subset=["text"])
context = new_df[["title", "number", "start", "end", "text"]].to_string(index=False)

prompt = f"""
You are a teaching assistant for the course:
Python for Beginners (Full Course) | 100DaysOfCode Programming.

Below are the most relevant subtitle chunks from the course.
Each chunk contains the video title, video number, start time, end time, and transcript text.

COURSE CONTEXT:
{context}
-----------------------------------------------------------------------------------------------------------------------------------------------------------
USER QUESTION:
{incoming_query}

Answer the user's question using only the provided course context.

Also:
- Tell the user which video contains the relevant content.
- Mention the relevant timestamp(s).
- Briefly explain what is taught there.
- Guide the user to the appropriate video and timestamp.
- If the question is unrelated to the provided course content, say that you can only answer questions related to this course.
- Do not ask follow-up questions.
- Do not end the response with a question.
"""

response= inference(prompt)
for chunk in response:
    print(chunk['message']['content'], end='', flush=True)

# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# with open("response.json", "w") as f:   #For, stream=False
#     f.write(response.model_dump_json(indent=2))

# for row in new_df.itertuples():
#     print(f"{row.Index}  {row.title}  {row.number}  {row.text}  {row.start}  {row.end}")
