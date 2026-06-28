from models.audio import audio_expert

def check_audio_deepfake(audio_bytes: bytes):
    """Check if audio is deepfake or real"""
    return audio_expert.predict(audio_bytes)