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


if not sys.warnoptions:
    os.environ['PYTHONWARNINGS'] = 'ignore:ResourceWarning'

# Initialize OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up PyAudio
p = pyaudio.PyAudio()

# Set up the audio playback
pygame.mixer.init()

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
def get_response(text):
    with open("sysmsg.txt", "r") as file:
        system_message = file.read().strip()

    delay_time = 0.01
    max_response_length = 200
    response_text = ""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ],
        max_tokens=max_response_length,
        temperature=0,
        stream=True,
    )

    for event in response:
        event_text = event['choices'][0]['delta']
        content = event_text.get('content', '')
        response_text += content
        play_response(content)
        time.sleep(delay_time)

    return response_text.strip()


def play_audio_from_queue(audio_queue):
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=22050,
        output=True,
    )

    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break

        stream.write(audio_chunk)

    stream.stop_stream()
    stream.close()

def play_response(text):
    env_path = "/home/rcolman/mimic3/.venv"
    voice = "en_US/m-ailabs_low#mary_ann"

    audio_queue = Queue(maxsize=5)
    audio_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))
    audio_thread.start()

    mimic3_process = subprocess.Popen(
        [f"{env_path}/bin/mimic3", "--interactive", "--process-on-blank-line", "--voice", voice],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": env_path},
    )

    mimic3_process.stdin.write(text + "\n")
    mimic3_process.stdin.flush()



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

# Close Mimic3 when done
close_mimic3()

