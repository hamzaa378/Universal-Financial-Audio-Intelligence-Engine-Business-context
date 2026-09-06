from analysis_ai.acoustic import analyze_acoustics

def call_quality(audio,sr=16000):
    return analyze_acoustics(audio,sr)["quality"]
