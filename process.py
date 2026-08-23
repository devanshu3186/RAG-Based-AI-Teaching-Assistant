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
# ffmpeg -i video.mp4 audio.mp3
# ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
# ffmpeg -i input.mp3 -t 10 -c copy output.mp3
# subprocess.run(["ffmpeg", 
#                "-i", 
#                "audios/7_Exercise 1_ Calculator using Python.mp3", 
#                "-t", 
#                "10", 
#                "-c", 
#                "copy", 
#                "audios/sample.mp3"
# ])
