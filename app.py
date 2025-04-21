import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
from google.generativeai import GenerativeModel, configure
from train import Transformer, MelodyGenerator, MelodyPreprocessor
import scipy.io.wavfile
from render_music import piano_sound, SAMPLE_RATE, BEAT_DURATION
import time
from rank_bm25 import BM25Okapi
import os


BATCH_SIZE = 32
DATA_PATH = "dataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100

# Page config for a better looking app
st.set_page_config(
    page_title="AI Story & Music Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .main > div {
        padding: 2em;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 1em 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border: none;
        border-radius: 10px;
        font-size: 18px;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

# Configure Gemini
configure(api_key="AIzaSyDdi0yy3Ycs-pFe9XcbH1QJdcLyEh5HIoc")


def load_stories():
    """Load stories from train.txt file"""
    try:
        with open("train.txt", "r", encoding="utf-8") as file:
            stories = file.readlines()
        return [story.strip() for story in stories if story.strip()]
    except FileNotFoundError:
        st.warning("Warning: train.txt not found. Running without story retrieval.")
        return None

def setup_retrieval():
    """Setup BM25 retrieval system"""
    stories = load_stories()
    if stories:
        tokenized_docs = [doc.lower().split() for doc in stories]
        return BM25Okapi(tokenized_docs), stories
    return None, None

def retrieve_similar_stories(query, bm25, stories, top_k=3):
    """Retrieve similar stories using BM25"""
    if not bm25 or not stories:
        return []
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [stories[i] for i in top_indices]


# Initialize session state
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False

def setup_melody_generator():
    """Initialize and load the trained melody generator"""
    melody_preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=100,
        max_num_positions_in_pe_decoder=100,
        dropout_rate=0.1,
    )
    
    # Load trained weights if available
    try:
        transformer_model.load_weights("trained_weights/melody_model")
    except:
        st.warning("No pre-trained weights found. Using untrained model.")
    
    return MelodyGenerator(transformer_model, melody_preprocessor.tokenizer)

# def generate_story(query):
#     model = GenerativeModel("gemini-2.0-flash")
#     prompt = f"""Write an emotional and detailed story about: {query}
#     The story should be:
#     - Focused on the theme
#     - Rich in emotional detail
#     - Have a clear beginning, middle, and end
#     - Include vivid characters
#     - Be exactly 100 words"""
    
#     response = model.generate_content(prompt)
#     return response.text.strip()

def generate_story(query, bm25=None, stories=None):
    """Generate story using Gemini with BM25 retrieval"""
    model = GenerativeModel("gemini-2.0-flash")
    
    # Get similar stories using BM25
    similar_stories = retrieve_similar_stories(query, bm25, stories)
    
    if similar_stories:
        context = "\n".join(similar_stories)
        prompt = f"""Based on these similar stories as context:
{context}

Write an emotional and detailed story about: {query}
The story should be:
- Focused on the theme
- Rich in emotional detail
- Have a clear beginning, middle, and end
- Include vivid characters from the context itself only create if characters are not present in the context
- Be exactly 100 words
"""
    else:
        prompt = f"""Write an emotional and detailed story about: {query}
The story should be:
- Focused on the theme
- Rich in emotional detail
- Have a clear beginning, middle, and end
- Include vivid characters
- Be exactly 100 words"""
    
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_music_prompt(story):
    model = GenerativeModel("gemini-2.0-flash")
    prompt = f"""Create a detailed music prompt for this story: {story}
    Describe:
    - Musical style and genre (focus on nostalgic game music)
    - Key instruments and their emotional roles
    - Tempo and rhythm patterns
    - Mood and atmosphere
    - Special sound effects and textures
    - at max 100 words
    """
    
    response = model.generate_content(prompt)
    return f"A musical piece that sounds like {response.text.strip()}"

def render_piano_sequence(melody, sample_rate=SAMPLE_RATE):
    """Convert melody sequence to piano audio with proper padding"""
    audio = np.zeros(0, dtype=np.float32)  # Start with empty array
    note_to_midi = {'C4': 60, 'D4': 62, 'E4': 64, 'G4': 67}
    
    for note_str in melody.strip().split():
        try:
            note, duration = note_str.split('-')
            duration = float(duration) * BEAT_DURATION
            freq = 440 * 2 ** ((note_to_midi[note] - 69) / 12)
            
            # Generate note audio
            note_audio = piano_sound(freq, duration)
            
            # Ensure both arrays have compatible shapes
            if len(audio) == 0:
                audio = note_audio
            else:
                # Pad the shorter array if needed
                if len(audio) < len(note_audio):
                    audio = np.pad(audio, (0, len(note_audio) - len(audio)))
                elif len(audio) > len(note_audio):
                    note_audio = np.pad(note_audio, (0, len(audio) - len(note_audio)))
                audio = np.concatenate([audio, note_audio])
                
        except (ValueError, KeyError) as e:
            st.warning(f"Skipping invalid note: {note_str}")
            continue
    
    return (audio * 32767).astype(np.int16)


def create_note_to_midi_mapping():
    """Create a mapping of notes (including sharps and flats) to MIDI numbers for octaves 2-6."""
    base_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    sharp_to_flat = {
        'C#': 'Db',
        'D#': 'Eb',
        'F#': 'Gb',
        'G#': 'Ab',
        'A#': 'Bb'
    }
    note_to_midi = {}
    for octave in range(2, 7):  # Covers octaves 2-6
        for i, note in enumerate(base_notes):
            note_name = f"{note}{octave}"
            midi_number = (12 * (octave + 1)) + i
            note_to_midi[note_name] = midi_number
            # Add flat equivalents for sharps
            if note in sharp_to_flat:
                flat_name = f"{sharp_to_flat[note]}{octave}"
                note_to_midi[flat_name] = midi_number
    return note_to_midi


# def create_note_to_midi_mapping():
#     """Create a mapping of notes to MIDI numbers"""
#     base_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#     note_to_midi = {}
    
#     for octave in range(2, 6):  # Octaves 2-5
#         for i, note in enumerate(base_notes):
#             note_name = f"{note}{octave}"
#             midi_number = (12 * (octave + 1)) + i
#             note_to_midi[note_name] = midi_number
            
#             # Add flat equivalents
#             if '#' in note:
#                 flat_name = f"{chr(ord(note[0]) + 1)}b{octave}"
#                 note_to_midi[flat_name] = midi_number
    
#     return note_to_midi



# def render_piano_sequence(melody, sample_rate=SAMPLE_RATE):
#     """Convert melody sequence to piano audio with fixed length samples"""
#     audio = np.array([], dtype=np.float32)
#     note_to_midi = create_note_to_midi_mapping()
#     base_length = int(SAMPLE_RATE * BEAT_DURATION)  # Use this as standard length unit
    
#     for note_str in melody.strip().split():
#         try:
#             note, duration = note_str.split('-')
#             duration = float(duration)
            
#             if note in note_to_midi:
#                 freq = 440 * 2 ** ((note_to_midi[note] - 69) / 12)
                
#                 # Calculate exact number of samples needed
#                 desired_samples = int(duration * base_length)
                
#                 # Generate note with exact length
#                 t = np.linspace(0, duration * BEAT_DURATION, desired_samples)
#                 note_audio = 0.5 * np.sin(2 * np.pi * freq * t)
                
#                 # Apply envelope
#                 envelope = np.exp(-3 * t / (duration * BEAT_DURATION))
#                 note_audio = note_audio * envelope
                
#                 # Concatenate with existing audio
#                 if len(audio) == 0:
#                     audio = note_audio
#                 else:
#                     audio = np.concatenate([audio, note_audio])
#             else:
#                 st.warning(f"Skipping unknown note: {note}")
                
#         except (ValueError, KeyError) as e:
#             st.warning(f"Error processing note {note_str}: {str(e)}")
#             continue
    
#     if len(audio) > 0:
#         # Normalize and convert to 16-bit PCM
#         audio = audio / np.max(np.abs(audio))
#         return (audio * 32767).astype(np.int16)
#     else:
#         return np.zeros(1000, dtype=np.int16)  # Return silence if no notes were processed

# def render_piano_sequence(melody, sample_rate=SAMPLE_RATE):
#     """Convert melody sequence to piano audio with fixed length samples"""
#     audio = np.array([], dtype=np.float32)
#     note_to_midi = create_note_to_midi_mapping()
#     base_length = int(SAMPLE_RATE * BEAT_DURATION)  # Use this as standard length unit

#     for note_str in melody.strip().split():
#         try:
#             note, duration = note_str.split('-')
#             duration = float(duration)

#             if note in note_to_midi:
#                 freq = 440 * 2 ** ((note_to_midi[note] - 69) / 12)

#                 # Calculate exact number of samples needed
#                 desired_samples = int(duration * base_length)

#                 # Generate note with exact length
#                 t = np.linspace(0, duration * BEAT_DURATION, desired_samples)
#                 note_audio = 0.5 * np.sin(2 * np.pi * freq * t)

#                 # Apply envelope
#                 envelope = np.exp(-3 * t / (duration * BEAT_DURATION))
#                 note_audio = note_audio * envelope

#                 # Concatenate with existing audio
#                 if len(audio) == 0:
#                     audio = note_audio
#                 else:
#                     audio = np.concatenate([audio, note_audio])
#             else:
#                 st.warning(f"Skipping unknown note: {note}")

#         except (ValueError, KeyError) as e:
#             st.warning(f"Error processing note {note_str}: {str(e)}")
#             continue

#     if len(audio) > 0:
#         # Normalize and convert to 16-bit PCM
#         audio = audio / np.max(np.abs(audio))
#         return (audio * 32767).astype(np.int16)
#     else:
#         return np.zeros(1000, dtype=np.int16)  # Return silence if no notes were processed

def render_piano_sequence(melody, sample_rate=SAMPLE_RATE):
    """Convert melody sequence to piano audio with fixed length samples, supporting all notes in dataset."""
    import numpy as np
    audio = np.array([], dtype=np.float32)
    note_to_midi = create_note_to_midi_mapping()
    base_length = int(sample_rate * BEAT_DURATION)
    # Support both comma and space separated notes
    for note_str in melody.replace(',', ' ').split():
        note_str = note_str.strip()
        if not note_str:
            continue
        try:
            note, duration = note_str.split('-')
            duration = float(duration)
            if note in note_to_midi:
                freq = 440 * 2 ** ((note_to_midi[note] - 69) / 12)
                desired_samples = int(duration * base_length)
                t = np.linspace(0, duration * BEAT_DURATION, desired_samples)
                note_audio = 0.5 * np.sin(2 * np.pi * freq * t)
                envelope = np.exp(-3 * t / (duration * BEAT_DURATION))
                note_audio = note_audio * envelope
                if len(audio) == 0:
                    audio = note_audio
                else:
                    audio = np.concatenate([audio, note_audio])
            else:
                st.warning(f"Skipping unknown note: {note}")
        except (ValueError, KeyError) as e:
            st.warning(f"Error processing note {note_str}: {str(e)}")
            continue
    if len(audio) > 0:
        audio = audio / np.max(np.abs(audio))
        return (audio * 32767).astype(np.int16)
    else:
        return np.zeros(1000, dtype=np.int16)

def main():
    # Header
    st.title("Emotion Driven Story Telling with Genrated Music")

    
    bm25, stories = setup_retrieval()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150?text=Music+AI", caption="AI Composer")
        st.markdown("### About")
        st.write("""
        This app uses AI to create unique stories and matching music.
        1. Enter your theme
        2. Get a custom story
        3. Listen to AI-generated music
        """)
        
        if bm25 and stories:
            st.success("✅ Story retrieval system loaded")
        else:
            st.warning("⚠️ Running without story retrieval")
    
    # Main content
    query = st.text_input(
        "Enter your theme:",
        placeholder="Example: A magical forest adventure",
        help="Try to be specific about the mood and setting you want"
    )
    
    if st.button("🚀 Generate Story & Music", use_container_width=True):
        st.session_state.generation_complete = False
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Story Generation
        status_text.text("📝 Generating story...")
        progress_bar.progress(25)
        
        # Generate story without showing retrieved stories
        story = generate_story(query, bm25, stories)
        
        # Display only the generated story with animation
        st.markdown("### 📖 Generated Story")
        story_container = st.empty()
        for i in range(len(story)):
            story_container.markdown(f">{story[:i+1]}_")
            time.sleep(0.01)
        story_container.markdown(f">{story}")
        
        # Music Generation using train.py's sequence generation
        status_text.text("🎹 Composing melody...")
        progress_bar.progress(50)
        
        try:
            # ... rest of your generation code remains the same ...
            melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
            train_dataset = melody_preprocessor.create_training_dataset()
            vocab_size = melody_preprocessor.number_of_tokens_with_padding

            transformer_model = Transformer(
                num_layers=2,
                d_model=64,
                num_heads=2,
                d_feedforward=128,
                input_vocab_size=vocab_size,
                target_vocab_size=vocab_size,
                max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
                max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
                dropout_rate=0.1,
            )

            melody_generator = MelodyGenerator(
                transformer_model, 
                melody_preprocessor.tokenizer
            )
            
            start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
            new_melody = melody_generator.generate(start_sequence)
            
            # Generate piano sequence
            piano_audio = render_piano_sequence(new_melody, SAMPLE_RATE)
            scipy.io.wavfile.write("generated_melody.wav", SAMPLE_RATE, piano_audio)
            
            st.markdown("### 🎵 Generated Melody")
            st.audio("generated_melody.wav")
            
            status_text.text("🎵 Creating music description...")
            progress_bar.progress(75)
            music_prompt = generate_music_prompt(story)
            
            st.markdown("### 🎼 Music Description")
            st.write(music_prompt)
            
            # Final Music Generation
            status_text.text("🎼 Creating final composition...")
            progress_bar.progress(90)
            
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            inputs = processor(
                text=[music_prompt],
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=True)
            
                        
            audio_data = output[0].cpu().numpy().squeeze()
            audio_data = np.clip(audio_data / np.max(np.abs(audio_data)), -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            scipy.io.wavfile.write("final_music.wav", 32000, audio_data)
            
            st.markdown("### 🎵 Final Composition")
            st.audio("final_music.wav")
            
            progress_bar.progress(100)
            status_text.text("✨ Generation complete!")
            st.session_state.generation_complete = True
            
        except Exception as e:
            st.error(f"An error occurred during melody generation: {str(e)}")
            status_text.text("❌ Generation failed")
            progress_bar.empty()


       
if __name__ == "__main__":
    main()