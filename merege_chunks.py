import os
import json
import math

n=5
for filename in os.listdir("json"):
    if filename.endswith(".json"):
        file_path= os.path.join("json", filename)

        with open(file_path, "r") as f:
            data=json.load(f)
            new_chunks=[]
            num_chunks=len(data["chunks"])
            new_group=math.ceil(num_chunks/n)

            for i in range(new_group):
                start_idx = i*n
                end_idx= min((i+1)*n, num_chunks)
                chunks_group=data["chunks"][start_idx: end_idx]
                
                new_chunks.append({
                    "number": data["chunks"][0]["number"],
                    "title": chunks_group[0]["title"],
                    "start": chunks_group[0]["start"],
                    "end": chunks_group[-1]["end"],
                    "text": " ".join([c["text"] for c in chunks_group])
                })
            os.makedirs("new_jsons", exist_ok=True)
            with open(os.path.join("new_jsons", filename), "w") as json_file:
                json.dump({"chunks": new_chunks, "text": data["text"]}, json_file, indent=4)