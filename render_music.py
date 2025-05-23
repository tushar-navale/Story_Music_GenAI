import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Configuration
SAMPLE_RATE = 44100
BPM = 120
BEAT_DURATION = 60 / BPM  # seconds per beat

# Your melody (note-duration pairs)
melody = """
C4-1.0 D4-1.0 E4-1.0 C4-1.0 D4-1.0 E4-1.0 D4-1.0 D4-1.0 C4-1.0 D4-1.0 E4-1.0 D4-1.0 C4-1.0 D4-1.0 C4-1.0 D4-1.0 E4-1.0 E4-1.0 E4-1.0 D4-1.0 E4-1.0 D4-1.0 E4-1.0 D4-1.0 E4-1.0 D4-1.0 E4-1.0 D4-1.0 E4-1.0 E4-1.0
""".strip().split()

# Piano frequency map (MIDI note numbers)
note_to_midi = {
    # Octave 3
    'A3': 57, 'Bb3': 58, 'B3': 59,
    
    # Octave 4 (most common octave in your dataset)
    'C4': 60, 'C#4': 61, 'Db4': 61,
    'D4': 62, 'D#4': 63, 'Eb4': 63,
    'E4': 64, 'F4': 65, 'F#4': 66, 'Gb4': 66,
    'G4': 67, 'G#4': 68, 'Ab4': 68,
    'A4': 69, 'A#4': 70, 'Bb4': 70,
    'B4': 71,
    
    # Octave 5
    'C5': 72, 'C#5': 73, 'Db5': 73,
    'D5': 74, 'D#5': 75, 'Eb5': 75,
    'E5': 76, 'F5': 77, 'F#5': 78, 'Gb5': 78,
    'G5': 79, 'G#5': 80, 'Ab5': 80,
    'A5': 81, 'A#5': 82, 'Bb5': 82,
    'B5': 83
}

def piano_sound(freq, duration, volume=0.5):
    """Generate realistic piano sound with harmonics and envelope"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Piano harmonics (5 partials)
    wave = (
        0.60 * np.sin(2 * np.pi * 1 * freq * t) +  # Fundamental
        0.25 * np.sin(2 * np.pi * 2 * freq * t) +  # Octave
        0.10 * np.sin(2 * np.pi * 3 * freq * t) +  # Perfect fifth
        0.05 * np.sin(2 * np.pi * 4 * freq * t) +  # Octave
        0.03 * np.sin(2 * np.pi * 6 * freq * t)     # Fifth above octave
    )
    
    # ADSR envelope (Attack, Decay, Sustain, Release)
    envelope = np.ones_like(t)
    attack = int(0.03 * SAMPLE_RATE)  # 30ms attack
    decay = int(0.1 * SAMPLE_RATE)    # 100ms decay
    sustain_level = 0.6
    release = int(0.5 * SAMPLE_RATE)  # 500ms release
    
    # Build envelope
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    envelope[-release:] *= np.linspace(1, 0, release)
    
    # Add slight randomness for realism
    wave *= (0.98 + 0.04 * np.random.random(len(t)))
    
    return wave * envelope * volume

# Generate audio
audio = np.array([], dtype=np.float32)
for note_str in melody:
    try:
        note, duration = note_str.split('-')
        duration = float(duration) * BEAT_DURATION
        freq = 440 * 2 ** ((note_to_midi[note] - 69) / 12)  # MIDI to frequency
        note_audio = piano_sound(freq, duration)
        audio = np.concatenate((audio, note_audio))
    except ValueError:
        continue

# Normalize and convert to 16-bit
peak = np.max(np.abs(audio))
audio = (audio * (32767 / peak)).astype(np.int16)

# Save as WAV
write("piano_melody.wav", SAMPLE_RATE, audio)

# Plot first 2 seconds of audio
plt.figure(figsize=(12, 4))
plt.plot(audio[:2*SAMPLE_RATE])
plt.title("Waveform Preview (First 2 Seconds)")
plt.show()

print("Successfully created piano_melody.wav")
print(f"Duration: {len(audio)/SAMPLE_RATE:.2f} seconds")