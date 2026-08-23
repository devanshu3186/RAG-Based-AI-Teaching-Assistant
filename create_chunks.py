import whisper
import json
import os

model = whisper.load_model("large-v3")

audios=os.listdir("audios")

for audio in audios:
    if ("_" in audio):
        number, title=audio.split("_", 1)  #split only at first underscore
        title=os.path.splitext(title)[0]   #split the file name from its extension or last period (.) of the filename.
        print(number, title)

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        chunks=[]
        for segment in result["segments"]:
            chunks.append(
                {"number":number,
                 "title":title,
                "start":segment["start"], 
                "end":segment["end"], 
                "text":segment["text"]
                }
            )
        chunks_with_metadata= {"chunks":chunks, "text": result["text"]}
        
        with open(f"json/{audio}.json", "w") as json_file:
            json.dump(chunks_with_metadata, json_file, indent=4)