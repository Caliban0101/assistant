import time
import os
import openai
import pyaudio
import sys
import json
from vosk import Model, KaldiRecognizer
import tempfile
import subprocess
import pygame
from io import BytesIO

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
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message['content'].strip()

mimic3_process = None

def init_mimic3():
    global mimic3_process
    if mimic3_process is None:
        mimic3_process = subprocess.Popen(
            "mimic3 --voice en_US/m-ailabs_low#mary_ann --interactive",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True
        )

def close_mimic3():
    global mimic3_process
    if mimic3_process is not None:
        mimic3_process.terminate()
        mimic3_process = None

def play_response(text):
    init_mimic3()
    mimic3_process.stdin.write(text + "\n")
    mimic3_process.stdin.flush()

    wav_data = bytearray()
    while True:
        line = mimic3_process.stdout.readline()
        if line.startswith("WV:"):
            wav_data.extend(bytes.fromhex(line[3:].strip()))
        elif line.startswith("WV-END"):
            break
        else:
            print(f"Unexpected output: {line.strip()}")

            with BytesIO(wav_data) as f:
                pygame.mixer.music.load(f)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

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
