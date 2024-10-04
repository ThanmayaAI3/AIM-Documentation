import pandas as pd
import torch
import os
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
import librosa
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
from transformers import AutoTokenizer
from PIL import Image

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load CLAP model
clap_model = AutoModel.from_pretrained("laion/clap-htsat-fused")
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

# go through the csv_file
file_name = 'modified_blues_csv_file.csv'
music_category = 'blues'
data_set_file = pd.read_csv("part1/part1/"+file_name)

# Specifications for downloading files from yt_dlp
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'ffmpeg_location': r'C:\ffmpeg\bin\ffmpeg.exe',
    'outtmpl': '%(title)s.%(ext)s',  # Name the output file based on the video title
}


def compute_imsm(audio, text, image):
    image = Image.open(image)

    # CLIP embeddings (Image and Text)
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    clip_outputs = clip_model(**inputs)
    image_embeds = clip_outputs.image_embeds
    text_embeds_clip = clip_outputs.text_embeds

    audio, sr1 = librosa.load(audio, sr=None)

    # CLAP embeddings (Audio and Text)
    inputs_audio = clap_processor(audios=audio, text=text, return_tensors="pt", padding=True, max_length = 77, sampling_rate = 48000)
    clap_outputs = clap_model(**inputs_audio)
    audio_embeds = clap_outputs.audio_embeds
    text_embeds_clap = clap_outputs.text_embeds

    # Compute cosine similarities between embeddings
    cos_sim_clip = torch.nn.functional.cosine_similarity(image_embeds, text_embeds_clip)
    cos_sim_clap = torch.nn.functional.cosine_similarity(audio_embeds, text_embeds_clap)

    # IMSM Metric Calculation
    imsm_score = torch.matmul(cos_sim_clip, cos_sim_clap.T)
    print(f"IMSM Score: {imsm_score.item()}")
    return imsm_score.item()


# Following code is specific for the MelBench dataset 
data_set_file['imsm score'] = ''
for index, item in data_set_file.iterrows():
    audio = ''
    image = 'part1/part1/' + music_category + '/images/' + item['image_path']
    if (not (os.path.exists(image) and os.path.isfile(image))):
        print(image)
        print("Doesn't exist")
        continue


    try:
        with YoutubeDL(ydl_opts) as ydl:
            video = ydl.extract_info(item['youtube_video_id'], download=True)
            audio = ydl.prepare_filename(video)
            audio = audio.rsplit('.', 1)[0] + '.mp3'
            text = item['Annotation']
            image = 'part1/part1/' + music_category + '/images/' + item['image_path']
            audio = audio.rsplit('.', 1)[0] + '.mp3'
            item['imsm score'] = compute_imsm(audio, text, image)
    except DownloadError as e:
        print(f"Error downloading video: {e}")
        continue

data_set_file.to_csv('imsm_output_scores.csv')
