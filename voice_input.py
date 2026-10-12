import speech_recognition as sr

def extract_command(text, keyword="go to"):
    text = text.lower().strip()
    if keyword in text:
        return text.split(keyword, 1)[1].strip()
    return ""

def voice_input(keyword="go to", timeout=5, phrase_time_limit=10):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening for '{keyword} ...'")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return ""

    try:
        heard = recognizer.recognize_google(audio)
    except (sr.UnknownValueError, sr.RequestError):
        return ""

    print("Heard:", heard)
    return extract_command(heard, keyword)

destination = voice_input()
if destination:
    print("Going to:", destination)
else:
    print("No command heard.")