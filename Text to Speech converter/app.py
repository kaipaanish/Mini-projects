import streamlit as st
import pyttsx3
import os

# Setup TTS engine
def text_to_speech(text, voice_gender='Male', filename='tts_output.wav'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.8)
    
    voices = engine.getProperty('voices')
    
    # Select voice based on gender
    if voice_gender == 'Male':
        engine.setProperty('voice', voices[0].id)
    else:
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        else:
            st.warning("No female voice found on this system.")
    
    engine.save_to_file(text, filename)
    engine.runAndWait()
    
    return filename

# Streamlit app
st.title("🗣️ Text to Speech App (with pyttsx3)")

# Input text
text = st.text_area("Enter the text you want to convert to speech:")

# Voice selection
voice = st.radio("Select Voice:", ['Male', 'Female'])

# Generate button
if st.button("Generate Audio"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        output_file = text_to_speech(text, voice)
        st.success("Audio generated successfully!")
        # Show audio player
        audio_file = open(output_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
