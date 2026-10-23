import pyttsx3

# Initialize the engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) 
engine.setProperty('rate', 150)

# Define the text and the output file name
text = "please turn"
output_file = "SoundAssets/turn.mp3"  # You can also use 'output.wav'

# Save the voice to a file
engine.save_to_file(text, output_file)
engine.runAndWait()

# Run and wait for the process to finish


print(f"Audio successfully saved to {output_file}")