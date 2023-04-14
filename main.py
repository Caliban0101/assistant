import time
import os
import openai
import pyaudio
from gtts import gTTS
from pygame import mixer
from io import BytesIO
import sys
import json
from vosk import Model, KaldiRecognizer
import tempfile


if not sys.warnoptions:
    os.environ['PYTHONWARNINGS'] = 'ignore:ResourceWarning'

# Initialize OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up PyAudio
p = pyaudio.PyAudio()

# Set up the audio playback
mixer.init()

# Activation word
activation_word = "margaret"

# Vosk Model
model = Model("vosk-model-small-en-us-0.15")

# Function to listen for activation word
def listen_for_activation_word():
    recognizer = KaldiRecognizer(model, 48000)
    stream = p.open(rate=48000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1024)

    print("Listening for activation word...")

    while True:
        data = stream.read(1024)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get('text')
            if activation_word in text.lower():
                stream.stop_stream()
                stream.close()
                return True
    stream.stop_stream()
    stream.close()

# Function to transcribe speech to text
def transcribe_speech():
    recognizer = KaldiRecognizer(model, 48000)
    stream = p.open(rate=48000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1024)

    print("Listening for your question...")
    while True:
        data = stream.read(1024)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get('text')
            stream.stop_stream()
            stream.close()
            return text
    stream.stop_stream()
    stream.close()

# Function to use ChatCompletion and get response
def get_response(text):
    with open("sysmsg.txt", "r") as file:
        system_message = file.read().strip()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message['content'].strip()

# Function to convert text to speech and play it
def play_response(text):
    tts = gTTS(text=text, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        tts.save(f.name)
        mixer.music.load(f.name)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)

    os.remove(f.name)

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
