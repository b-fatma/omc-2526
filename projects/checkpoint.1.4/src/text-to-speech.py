from TTS.api import TTS

# Text to convert
text = "Hello Open Minds! Get ready — this project is your first step into the world of multimedia systems, where images, audio, and artificial intelligence come together to do some pretty cool things."

# Initialize TTS with a pre-trained model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Save audio to file
tts.tts_to_file(text=text, file_path="audio.wav")

print("✅ Audio successfully saved as sample_audio.wav")