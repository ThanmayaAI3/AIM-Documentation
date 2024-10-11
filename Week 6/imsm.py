import pandas as pd
import torch
import os

import librosa
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
from PIL import Image


# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load CLAP model
clap_model = AutoModel.from_pretrained("laion/clap-htsat-fused")
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")


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

    # CLIP Metric Calculation
    print(f"CLIP Score: {cos_sim_clip.item()}")

    # CLAP Metric Calculation
    print(f"CLAP Score: {cos_sim_clap.item()}")

def compute_imsm_melfusion(image1, image2, text1, text2, audio1, audio2):
    # Load images
    image_list = [Image.open(image1), Image.open(image2)]

    input_text = [text1, text2]
    # Process text and image inputs with CLIP processor
    inputs = clip_processor(text=input_text, images=image_list, return_tensors="pt", padding=True)

    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs_clip = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    # Load audio files
    y1, sr1 = librosa.load(audio1, sr=None)
    y2, sr2 = librosa.load(audio2, sr=None)

    audio_sample = [y1, y2]

    # Process text and audio inputs with CLAP processor
    inputs = clap_processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

    outputs = clap_model(**inputs)
    logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
    probs_clap = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities

    # Calculate IMSM score
    probs_metric = probs_clip @ probs_clap.T
    imsm_score = probs_metric.softmax(dim=-1)

    print("CLAP score: ", probs_clap)
    print("CLIP score: ",probs_clip)
    print("IMSM Score:", imsm_score)

    # Print CLIP and CLAP scores
    print("CLIP Scores (Image-Text Similarity):")
    clip_scores = logits_per_image.softmax(dim=-1).detach().numpy()
    for i, row in enumerate(clip_scores):
        for j, score in enumerate(row):
            print(f"CLIP Score between image {i + 1} and text {j + 1}: {score:.4f}")

    print("\nCLAP Scores (Audio-Text Similarity):")
    clap_scores = logits_per_audio.softmax(dim=-1).detach().numpy()
    for i, row in enumerate(clap_scores):
        for j, score in enumerate(row):
            print(f"CLAP Score between audio {i + 1} and text {j + 1}: {score:.4f}")

    # Convert the tensor to a NumPy array for readability
    imsm_score_numpy = imsm_score.detach().numpy()

    # Print IMSM scores with numerical labels
    print("\nIMSM Scores (Image-Audio Similarity):")
    for i, row in enumerate(imsm_score_numpy):
        for j, score in enumerate(row):
            print(f"IMSM Score between image {i + 1} and audio {j + 1}: {score:.4f}")

def compute_imsm_melfusion_single(image, text, audio):
    # Load the image
    image_input = Image.open(image)

    # Process the text and image inputs with CLIP processor
    inputs = clip_processor(text=[text], images=[image_input], return_tensors="pt", padding=True)

    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs_clip = logits_per_image  # use logits directly without applying softmax

    # Load the audio file
    y, sr = librosa.load(audio, sr=None)
    audio_input = [y]

    # Process the text and audio inputs with CLAP processor
    inputs = clap_processor(text=[text], audios=audio_input, return_tensors="pt", padding=True)

    outputs = clap_model(**inputs)
    logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
    probs_clap = logits_per_audio  # use logits directly without applying softmax

    # Calculate IMSM score by combining the image-text and audio-text similarities
    imsm_score = probs_clip @ probs_clap.T  # combining logits without applying softmax

    print("CLAP score: ", probs_clap)
    print("CLIP score: ". probs_clip)
    print("IMSM Score:", imsm_score)


text_river = "soothing ambience of flowing water and a forest creek, making it ideal for relaxation, focus, meditation, or sleep."
image_river = "river.png"
audio_river = "20 Minutes of Relaxing River Sounds - Flowing Water and Forest Creek Ambience 🏞️.mp3"
audio_piano = "Yiruma - River Flows in You.mp3"
text_piano = "a soft, flowing piano composition with a gentle, romantic feel. The melody is simple yet deeply emotive, creating a tranquil and introspective atmosphere."
image_bear = "bear.jpg"
image_piano = "piano.jpg"

'''
print("All river data")
compute_imsm(audio_river, text_river, image_river)

print("All piano data")
compute_imsm(audio_piano, text_piano, image_piano)

print("three different sources")
compute_imsm(audio_river, text_piano, image_bear)
'''

### testing out melfusion code on the data that is specified above to see the difference (and hope for more reasonable scores)
default_text = "testing with the same text and seeing what the output is"
compute_imsm_melfusion(image_piano, image_river, text_piano, text_river, audio_piano, audio_river)

