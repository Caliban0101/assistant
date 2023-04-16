import time
import os
import threading
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
from pydub.playback import play
import asyncio
from pydub.playback import _play_with_simpleaudio as play_async
from queue import Queue

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

audio_queue = Queue()

def listen_for_keyboard_input():
    while True:
        question = input("Type your question: ")
        if question:
            loop_keyboard.run_until_complete(handle_question(question))



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


def audio_player():
    current_audio = None

    while True:
        next_audio = audio_queue.get()
        if next_audio is None:
            break

        if current_audio is None:
            current_audio = next_audio
        else:
            current_audio += next_audio

        play(current_audio)
        current_audio = None  # Reset current_audio for the next playback cycle


# Function to use ChatCompletion and get response
conversation_history = []


async def get_response(text):
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
        messages=conversation_history,
        stream=True
    )

    partial_sentence = ""
    for chunk in response:
        delta = chunk.get("choices")[0].get("delta", {})
        content = delta.get("content")
        if content is not None:
            partial_sentence += content  # Directly append the content without stripping or adding space

            if '.' in partial_sentence:
                sentences = partial_sentence.split('.')
                for sentence in sentences[:-1]:
                    yield clean_text(sentence) + '.'  # Use the clean_text function here

                partial_sentence = sentences[-1]

    if partial_sentence:
        yield clean_text(partial_sentence)  # Use the clean_text function here



def clean_text(text):
    cleaned_text = text.replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";").replace(" -", "-")
    return cleaned_text



async def play_response(sentence_generator):
    voice = "en_US/vctk_low#p238"

    # Start the player thread
    player_thread = threading.Thread(target=audio_player)
    player_thread.start()

    # Process each sentence
    async for sentence in sentence_generator:
        response = requests.post(
            f"{mimic3_server_url}/api/tts?voice={voice}",
            data=sentence.encode(),
            headers={"Content-Type": "text/plain"},
        )
        print("sent: " + sentence)
        if response.status_code == 200:
            wav_data = BytesIO(response.content)

            audio_segment = AudioSegment.from_file(wav_data, format="wav")
            audio_queue.put(audio_segment)
        else:
            print(f"Error: Mimic3 server returned status code {response.status_code}")

    # Stop the player thread after all audio segments have been processed
    audio_queue.put(None)
    player_thread.join()


async def handle_question(question):
    print("You asked:", question)
    sentence_generator = get_response(question)
    await play_response(sentence_generator)


# Main loop
async def main_loop():
    while True:
        if listen_for_activation_word():
            question = transcribe_speech()
            if question:
                print("You asked:", question)
                sentence_generator = get_response(question)
                await play_response(sentence_generator)
            else:
                print("Could not understand your question")



if __name__ == "__main__":
    # Create event loops for voice commands and keyboard input
    loop_voice = asyncio.new_event_loop()
    loop_keyboard = asyncio.new_event_loop()

    # Start the main loop for voice commands in a new thread
    main_loop_thread = threading.Thread(target=loop_voice.run_until_complete, args=(main_loop(),))
    main_loop_thread.start()

    # Start the keyboard input listening in a new thread
    keyboard_input_thread = threading.Thread(target=listen_for_keyboard_input)
    keyboard_input_thread.start()

    # Wait for the threads to finish
    main_loop_thread.join()
    keyboard_input_thread.join()