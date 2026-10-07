from sentence_transformers import util
import numpy as np
import joblib
from preprocess_json import create_embeddings
from ollama import chat
from openai import OpenAI


def inference(prompt):
    stream = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': prompt}],
    stream=True,
    think=False
    )

    return stream

def inference_openai(prompt):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return response

# Loading the saved DataFrame
df=joblib.load("chunks_embedding.pkl")

incoming_query=input("Ask a Question: ")
question_embedding=create_embeddings(incoming_query)

# Find similarities of question embedding with other embeddings 
chunk_embeddings_array = np.stack(df["embedding"].values) 
similarities = util.cos_sim(question_embedding, chunk_embeddings_array).flatten()

top_results=5
max_idx=similarities.argsort(descending=True)[:top_results].tolist()

new_df= df.loc[max_idx]
new_df= new_df.drop_duplicates(subset=["text"])
context = new_df[["title", "number", "start", "end", "text"]].to_string(index=False)

prompt = f"""
You are a teaching assistant for the course:

**Python for Beginners (Full Course) | 100DaysOfCode Programming**

Your job is to answer the user's question using **only the provided course context** and to respond according to **what the user is asking for**.

The course context consists of relevant subtitle chunks retrieved from the course. Each chunk contains:

* Video title
* Video number
* Start time
* End time
* Transcript text

## COURSE CONTEXT

{context}

## USER QUESTION

{incoming_query}

# RESPONSE MODE

First, determine what the user is asking for. Then choose the appropriate response mode.

## MODE 1 — REFERENCE ONLY

Use this mode when the user is asking only for the **location of a topic in the course**, such as:

* "What timestamp is string slicing?"
* "Which video teaches loops?"
* "Where is list comprehension explained?"
* "When does the instructor teach functions?"
* "Give me the timestamp for dictionaries."
* "Which video covers this topic?"

In this mode:

* Give only the relevant course reference.
* Do not explain or teach the concept unless the user also asks for an explanation.
* Do not add unnecessary information.

Provide:

**Video:** [Relevant video title / number]
**Timestamp:** [Relevant timestamp]

When useful, you may briefly identify the topic, but keep the response focused on the requested reference.

---

## MODE 2 — EXPLANATION + REFERENCE

Use this mode when the user asks to **explain, teach, define, clarify, understand, or learn** something.

Examples:

* "What is string slicing?"
* "Explain string slicing."
* "Can you explain loops?"
* "What does a dictionary do?"
* "Help me understand functions."
* "Explain this concept to me."

In this mode:

### First — Answer the question

Explain the concept clearly and in a beginner-friendly way using **only the provided course context**.

The explanation is the main answer.

Do not merely tell the user which video to watch.

### Then — Provide the course reference

After explaining the concept, provide the relevant course location:

**Video:** [Relevant video title / number]
**Timestamp:** [Relevant timestamp]
**What is taught:** [Brief description of the relevant section]

The course reference is supplementary to the explanation.

---

## MODE 3 — COMBINED REQUESTS

If the user explicitly asks for both an explanation and a course reference, provide both.

For example:

> "Explain string slicing and tell me where it is taught."

In this case:

1. Explain string slicing.
2. Give the relevant video.
3. Give the relevant timestamp.

---

# USING THE COURSE CONTEXT

Treat the provided course context as the **only source of truth**.

* Do not use outside knowledge to fill gaps.
* Do not introduce facts that are not supported by the provided course context.
* You may combine information from multiple relevant chunks when necessary.
* When several chunks are relevant, provide the most relevant course references.
* Do not invent video titles, video numbers, timestamps, or course content.

# MISSING OR INSUFFICIENT CONTEXT

If the provided course context is missing, empty, irrelevant, or does not contain enough information to answer the user's request:

* Do not guess.
* Do not use general knowledge to fill the gap.
* Do not fabricate an explanation.
* Do not invent a video title, video number, or timestamp.
* Clearly state that the available course context does not contain enough information to answer the user's question.

# UNRELATED QUESTIONS

If the user's question is unrelated to the provided course content, state that you can only answer questions related to this course.

Do not answer unrelated questions using outside knowledge.

# RESPONSE STYLE

* Be clear, concise, and helpful.
* Use simple, beginner-friendly language appropriate for a Python beginner.
* Match the amount of detail to what the user asked for.
* Do not overwhelm the user with unnecessary information.
* Do not ask follow-up questions.
* Do not end the response with a question.

# MOST IMPORTANT RULE

**Respond to what the user actually asked for.**

* If they ask only **where/when/which video/timestamp**, give the **reference only**.
* If they ask to **explain/teach/clarify/understand**, give the **explanation first, followed by the relevant course reference**.
* If they explicitly ask for **both**, provide both.
"""

# response= inference(prompt)
# for chunk in response:
#     print(chunk['message']['content'], end='', flush=True)

response=inference_openai(prompt)
print(response.output_text)

# with open("response.json", "w") as f:   #For, stream=False
#     f.write(response.model_dump_json(indent=2))

# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# for row in new_df.itertuples():
#     print(f"{row.Index}  {row.title}  {row.number}  {row.text}  {row.start}  {row.end}")
