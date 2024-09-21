from clap import CLAPModel
import torch
import torchaudio

# Load the CLAP model (pretrained)
model = CLAPModel.from_pretrained("clap-base")  # Adjust the model version if needed
model.eval()

# Load the audio file
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

# Encode audio and text to their respective embeddings
def get_similarity_score(audio_path, text_input):
    # Load and process audio
    waveform, sample_rate = load_audio(audio_path)
    audio_embedding = model.encode_audio(waveform, sample_rate)

    # Encode text input
    text_embedding = model.encode_text([text_input])

    # Compute cosine similarity between the two embeddings
    similarity = torch.nn.functional.cosine_similarity(audio_embedding, text_embedding)
    return similarity.item()

# Usage
audio_path = ""
text_input = "A person playing the piano"

score = get_similarity_score(audio_path, text_input)
print(f"Similarity score between the audio and text: {score}")
