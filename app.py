import os
import whisper
import yt_dlp
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from transformers import pipeline
import streamlit as st
from detoxify import Detoxify
import shutil
import uuid

# Initialize models
sentiment_model = pipeline("sentiment-analysis")
kw_model = KeyBERT()
whisper_model = whisper.load_model("small")
toxicity_model = Detoxify('original')

# Functions
def download_youtube_video(url):
    uid = str(uuid.uuid4())[:8]
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'video_{uid}.%(ext)s',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    for file in os.listdir():
        if file.startswith(f"video_{uid}") and file.endswith((".mp4", ".mkv", ".webm")):
            return file
    raise FileNotFoundError("Download failed.")

def extract_audio(video_path, output_audio="audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio, codec='pcm_s16le')
    return output_audio

def transcribe_audio(audio_path, language="en"):
    result = whisper_model.transcribe(audio_path, language=language)
    return result["text"]

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def analyze_sentiment(text):
    return sentiment_model(text[:512])[0]

def extract_keywords(text):
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)

def generate_chapters(text, duration, num_chapters=5):
    words = text.split()
    parts = np.array_split(words, num_chapters)
    timestamps = [round(i * duration / num_chapters, 2) for i in range(num_chapters)]
    return [{"time": t, "summary": " ".join(p[:15]) + "..."} for t, p in zip(timestamps, parts)]

def extract_scene_frames(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    prev, count = None, 0
    saved = []
    os.makedirs("scene_frames", exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            score = diff.mean()
            if score > threshold:
                path = f"scene_frames/frame_{count}.jpg"
                cv2.imwrite(path, frame)
                saved.append(path)
        prev = gray
        count += 1
    cap.release()
    return saved

def detect_toxicity(text):
    scores = toxicity_model.predict(text)
    flagged = {k: v for k, v in scores.items() if v > 0.5}
    return flagged if flagged else "No major toxicity/controversy detected."

def generate_blog_or_tweet(transcript, keywords, mode="blog"):
    key_phrases = ", ".join([kw[0] for kw in keywords])
    if mode == "blog":
        return f"""📝 **Auto Blog Summary**

**Overview:**
{transcript[:300]}...

**Key Topics:**
{key_phrases}

**Conclusion:**
This video dives into {keywords[0][0]}, exploring insights and takeaways you won’t want to miss!

#AI #Summarizer #VideoToBlog
"""
    else:
        tweets = [
            "🧵 Thread: Summary of a fascinating video!",
            f"1. Key Theme: {keywords[0][0]}",
            f"2. Other Highlights: {', '.join([kw[0] for kw in keywords[1:4]])}",
            f"3. Summary Snippet: {transcript[:200]}...",
            "4. My Takeaway: Super informative and relevant!",
            "#AI #VideoSummary #ContentToThread"
        ]
        return "\n\n".join(tweets)

# Streamlit App
st.set_page_config(page_title="🎬 SummarAIze", layout="wide")
st.title("🎥 SummarAIze - YouTube Video Summarizer")

video_url = st.text_input("Enter YouTube Video URL")
trigger = st.button("Summarize Video 🎬")

if trigger and video_url:
    # Clean old files
    for f in os.listdir():
        if f.startswith("video_") or f.endswith((".wav", ".jpg")):
            os.remove(f)
    if os.path.exists("scene_frames"):
        shutil.rmtree("scene_frames")

    with st.spinner("Downloading video..."):
        video_file = download_youtube_video(video_url)

    with st.spinner("Extracting audio..."):
        audio_file = extract_audio(video_file)

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(audio_file)

    st.subheader("📄 Transcript")
    st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)

    st.subheader("🔍 Keywords")
    keywords = extract_keywords(transcript)
    st.write(keywords)

    st.subheader("😃 Emotion Analysis")
    st.write(analyze_sentiment(transcript))

    st.subheader("☢️ Toxicity / Controversy Detection")
    st.write(detect_toxicity(transcript))

    st.subheader("📝 Auto-Generated Blog")
    blog = generate_blog_or_tweet(transcript, keywords, mode="blog")
    st.markdown(blog)

    st.subheader("🐦 Twitter Thread Summary")
    tweet = generate_blog_or_tweet(transcript, keywords, mode="tweet")
    st.markdown(tweet)

    st.subheader("🌐 Translations")
    for lang in ["en", "hi", "te"]:
        st.markdown(f"**{lang.upper()}**")
        st.info(translate_text(transcript, lang))

    st.subheader("🕐 Timeline Chapters")
    video_clip = VideoFileClip(video_file)
    chapters = generate_chapters(transcript, video_clip.duration)
    for ch in chapters:
        st.markdown(f"⏱️ `{ch['time']}s`: {ch['summary']}")

    st.subheader("🖼️ Key Scene Frames")
    frames = extract_scene_frames(video_file)
    for f in frames[:10]:
        st.image(f, width=350)
