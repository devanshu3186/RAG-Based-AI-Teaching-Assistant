# convert the videos to mp3
import os
import subprocess

files=os.listdir("videos")
for file in files:
    tutorial_num=file.split(" #")[1].split(" ")[0]
    file_name=file.split(".com ")[1].split(" _ ")[0]
    subprocess.run([
        "ffmpeg", 
        "-i", f"videos/{file}", 
        "-vn", 
        "-ac", "1", 
        "-ar", "16000", 
        f"audios/{tutorial_num}_{file_name}.mp3"
])

