import time
import os
import openai
import pyaudio
import speech_recognition as sr
from gtts import gTTS
from pygame import mixer
from io import BytesIO
import sys

if not sys.warnoptions:
        os.environ['PYTHONWARNINGS'] = 'ignore:ResourceWarning'

# Initialize OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up PyAudio
p = pyaudio.PyAudio()

# Set up speech recognition
r = sr.Recognizer()

# Set up the audio playback
mixer.init()

# Activation word
activation_word = "margaret"

# Function to listen for activation word
def listen_for_activation_word():
        pa = pyaudio.PyAudio()
        stream = pa.open(
                rate=48000,
                channels=1,
                format=pyaudio.paInt24,
                input=True,
                frames_per_buffer=1024,
        )

        r = sr.Recognizer()
    with sr.AudioSource(stream) as source:
        print("Listening for activation word...")
        r.pause_threshold = 1.0
	r.adjus_for_ambient_noise(source, duration=3)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            if activation_word in text.lower():
                return True
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
    return False
    stream.stop_stream()
    stream.close()
    pa.terminate()

# Function to transcribe speech to text
def transcribe_speech():
        pa = pyaudio.PyAudio()
        stream = pa.open(
                rate=48000,
                channels=1,
                format=pyaudio.paInt24,
                input=True,
                Frames_per_buffer=1024,
        )
        r = sr.Recognizer()
    with sr.AudioSource(stream) as source:
        print("Listening for your question...")
        r.pause_threshold = 1.0
        r.adjust_for_ambient_noise(source, duration=3)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
    return None
    stream.stop_stream()
    stream.close()
    pa.terminate()


# Function to use ChatCompletion and get response
def get_response(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].content.strip()

# Function to convert text to speech and play it
def play_response(text):
    tts = gTTS(text=text, lang='en')
    with BytesIO() as f:
        tts.save(f)
        f.seek(0)
        mixer.music.load(f)
        mixer.music.play()

# Main loop
while True:
    if listen_for_activation_word():
        question = transcribe_speech()
        if question:
            print("You asked:", question)
            response_text = get_response(question)
            print("Assistant:", response_text)
            play_response(response_text)
        else:
            print("Could not understand your question")
