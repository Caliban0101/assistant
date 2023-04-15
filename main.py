import time
import os
import threading
from queue import Queue
import openai
import pyaudio
import sys
import json
from vosk import Model, KaldiRecognizer
import tempfile
import subprocess
import pygame
from io import BytesIO
import wave
import requests
from tempfile import NamedTemporaryFile
import nltk
from pydub import AudioSegment

if not sys.warnoptions:
    os.environ['PYTHONWARNINGS'] = 'ignore:ResourceWarning'

# Initialize OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up PyAudio
p = pyaudio.PyAudio()

# Set up the audio playback
pygame.mixer.init()

# Activation word
activation_word = "minerva"

# Vosk Model
model = Model("vosk-model-small-en-us-0.15")

# Mimic3 server URL
mimic3_server_url = "http://0.0.0.0:59125"

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
                # Play the ding sound
                ding_sound = "ping.mp3"  # Replace with your file name
                pygame.mixer.music.load(ding_sound)
                pygame.mixer.music.play()

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
conversation_history = []


def get_response(text):
    global conversation_history

    with open("sysmsg.txt", "r") as file:
        system_message = file.read().strip()

    if not conversation_history:
        conversation_history.append({"role": "system", "content": system_message})

    conversation_history.append({"role": "user", "content": text})

    while sum(len(msg["content"]) for msg in conversation_history) > 3000:
        conversation_history.pop(0)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    reply = response.choices[0].message['content'].strip()
    conversation_history.append({"role": "assistant", "content": reply})

    return reply

def play_response(text):
    voice = "en_US/vctk_low#p264"

    # Split the input text into individual lines or sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # Process each sentence
    combined_audio = None
    for sentence in sentences:
        # Create a temporary file to store the current sentence
        with NamedTemporaryFile(delete=False, mode="w+") as temp_file:
            temp_file.write(sentence)
            temp_file.flush()

            # Send a request to the Mimic3 server
            with open(temp_file.name, "rb") as file_data:
                response = requests.post(
                    f"{mimic3_server_url}/api/tts?voice={voice}",
                    data=file_data,
                    headers={"Content-Type": "text/plain"},
                )

            # Remove the temporary file
            os.unlink(temp_file.name)

            if response.status_code == 200:
                wav_data = BytesIO(response.content)

                # Pass the wav_data directly to AudioSegment.from_file
                audio_segment = AudioSegment.from_file(wav_data, format="wav")

                if combined_audio is None:
                    combined_audio = audio_segment
                else:
                    combined_audio += audio_segment
            else:
                print(f"Error: Mimic3 server returned status code {response.status_code}")

    # Play the combined audio
    if combined_audio:
        play(combined_audio)
    else:
        print("No audio data to play.")


# Function to play the audio using pydub
def play(audio_segment):
    from pydub.playback import play as playback

    playback(audio_segment)

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
