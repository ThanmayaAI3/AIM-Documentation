from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

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


with YoutubeDL(ydl_opts) as ydl:
    video = ydl.extract_info("https://www.youtube.com/watch?v=NPBCbTZWnq0", download=True)
    audio = ydl.prepare_filename(video)
