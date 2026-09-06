from analysis_ai.acoustic import analyze_acoustics

def detect_stress_markers(audio,sr=16000):
    return analyze_acoustics(audio,sr)["stress_markers"]
