import numpy as np
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import librosa
from render_music import piano_sound, SAMPLE_RATE, BEAT_DURATION



def load_stories():
    """Load stories from train.txt file"""
    try:
        with open("train.txt", "r", encoding="utf-8") as file:
            stories = file.readlines()
        return [story.strip() for story in stories if story.strip()]
    except FileNotFoundError:
        print("Warning: train.txt not found. Running without BM25 retrieval.")
        return None

stories = load_stories()
# 2. Build BM25 Retrieval
tokenized_docs = [doc.lower().split() for doc in stories]
bm25 = BM25Okapi(tokenized_docs)

def retrieve_documents_bm25(query, top_k=3):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [stories[i] for i in top_indices]



from google.generativeai import GenerativeModel, configure 

configure(api_key="AIzaSyDdi0yy3Ycs-pFe9XcbH1QJdcLyEh5HIoc")  # Replace with your actual key

def gemini_generate(prompt ):
    #client = genai.Client(api_key="AIzaSyDdi0yy3Ycs-pFe9XcbH1QJdcLyEh5HIoc")  # Or use env var
    model = GenerativeModel("gemini-2.0-flash")
    
    # Generate the content
    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
        }
    )
    return response.text.strip()

def generate_story(query, stories_db=retrieve_documents_bm25):
    if stories_db:
        retrieved_docs = stories_db(query)
        context = " ".join(retrieved_docs)
        prompt = f"""Based on these similar stories as context:
{context}

Write an emotional and detailed story about: {query}
The story should be engaging and complete, focusing on the theme and emotions."""
    else:
        prompt = f"""Write an emotional and detailed story about: {query}
The story should be:
- Focused on the theme
- Rich in emotional detail
- Have a clear beginning, middle, and end
- Be also add characters
- and make the story with only 100 words

Begin the story:"""
    return gemini_generate(prompt)


from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

def setup_finetuned_music_model():
    """Set up the fine-tuned Unsloth model for music prompt generation"""
    # Load the saved fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "E:\6sem\GenAI\HandsOn\unit4\lora_model",  # Path to your saved model from the notebook
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Set up chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1"
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_music_prompt_finetuned(query, model, tokenizer):
    """Generate music prompt using fine-tuned model"""
    messages = [{
        "role": "user",
        "content": f"""Based on the theme: {query}
        Generate a detailed music description including:
        - Musical genre and style
        - Primary instruments and their roles
        - Tempo and rhythm characteristics
        - Emotional atmosphere and mood
        - Sound design elements and effects
        Make it suitable for nostalgic game music."""
    }]
    
    # Prepare input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate with tuned parameters
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=1.5,
        min_p=0.1,
        use_cache=True
    )
    
    # Process output
    generated_text = tokenizer.batch_decode(outputs)[0]
    response = generated_text.split("assistant")[1].strip()
    return f"A musical piece that captures: {response}"



def generate_music_prompt(query):
    """Generate a music prompt using Gemini"""
    prompt = f"""Generate a detailed music prompt that describes the sound, mood, instruments, and style for: {query}
The music should capture the essence of the story's emotions and atmosphere. 
Include specific musical elements like tempo, key instruments, and overall mood."""
    
    music_prompt = gemini_generate(prompt)
    return f"A musical piece that sounds like {music_prompt}."


def render_piano_sequence(melody):
    """Render a piano sequence using the melody"""
    audio = np.array([], dtype=np.float32)
    note_to_midi = {'C4': 60, 'D4': 62, 'E4': 64, 'G4': 67}
    
    for note_str in melody.strip().split():
        try:
            note, duration = note_str.split('-')
            duration = float(duration) * BEAT_DURATION
            freq = 440 * 2 ** ((note_to_midi[note] - 69) / 12)
            note_audio = piano_sound(freq, duration)
            audio = np.concatenate((audio, note_audio))
        except (ValueError, KeyError):
            continue
    
    peak = np.max(np.abs(audio))
    audio = (audio * (32767 / peak)).astype(np.int16)
    return audio


# def generate_final_music(piano_audio, music_prompt):
#     """Generate final music using MusicGen based on piano audio and prompt"""
#     try:
#         # Setup MusicGen
#         processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
#         model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
#         if torch.cuda.is_available():
#             model = model.to("cuda")
        
#         # Process with MusicGen - text only input
#         inputs = processor(
#             text=[music_prompt],
#             padding=True,
#             return_tensors="pt"
#         )
        
#         if torch.cuda.is_available():
#             inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
#         # Generate audio with adjusted parameters
#         with torch.no_grad():
#             output = model.generate(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=256,
#                 guidance_scale=3.0,
#                 do_sample=True
#             )
        
#         # Convert output to proper audio format
#         audio_data = output[0].cpu().numpy()
        
#         # Normalize and convert to 16-bit PCM
#         audio_data = np.clip(audio_data, -1.0, 1.0)
#         audio_data = (audio_data * 32767).astype(np.int16)
        
#         return audio_data, 32000  # MusicGen outputs at 32kHz
        
#     except Exception as e:
#         print(f"Error during music generation: {str(e)}")
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         raise

import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write as write_wav

def generate_final_music(piano_audio, music_prompt, output_path="generated_music.wav"):
    """Generate final music using MusicGen based on piano audio and prompt"""
    try:
        # Setup MusicGen
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        # Process with MusicGen - text only input
        inputs = processor(
            text=[music_prompt],
            padding=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate audio with adjusted parameters
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                guidance_scale=3.0,
                do_sample=True
            )
        
        # Convert output to proper audio format
        audio_data = output[0].cpu().numpy().squeeze()
        
        # Normalize audio to 16-bit PCM range
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save the audio file
        write_wav(output_path, 32000, audio_data)
        
        return audio_data, 32000  # Return audio data and sample rate
        
    except Exception as e:
        print(f"Error during music generation: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def main():
    # Get user input
    query = input("Enter a theme for your nostalgic story and music: ")
    
    # Setup generators
    #story_generator = setup_story_generation()
    #prompt_generator = setup_music_prompt_generation()
    
    # Generate story
    print("\nGenerating story...")
    story = generate_story(query)
    print(f"Generated Story:\n{story}\n")
    
    # Generate music prompt
    print("Generating music prompt...")
    music_prompt = generate_music_prompt(story)
    print(f"Music Prompt: {music_prompt}\n")
    
    # Define and render piano sequence
    melody = """
    C4-1.0 D4-1.0 E4-1.0 C4-1.0 D4-1.0 E4-1.0 D4-1.0 D4-1.0
    C4-1.0 D4-1.0 E4-1.0 D4-1.0 C4-1.0 D4-1.0 C4-1.0
    """
    print("Rendering piano sequence...")
    piano_audio = render_piano_sequence(melody)
    scipy.io.wavfile.write("piano_sequence.wav", SAMPLE_RATE, piano_audio)
    print("Piano sequence saved as 'piano_sequence.wav'\n")
    
    # Generate final music
    print("Generating final music with MusicGen...")
    # final_audio, sample_rate = generate_final_music(piano_audio, music_prompt)
    # scipy.io.wavfile.write("final_music.wav", sample_rate, final_audio)
    # print("Final music saved as 'final_music.wav'")
    # try:
    #     final_audio, sample_rate = generate_final_music(piano_audio, music_prompt)
        
    #     # Ensure the audio data is in the correct format
    #     if final_audio.dtype != np.int16:
    #         final_audio = np.int16(np.clip(final_audio, -32768, 32767))
        
    #     # Save as WAV file
    #     scipy.io.wavfile.write(
    #         "final_music.wav", 
    #         sample_rate, 
    #         final_audio
    #     )
    #     print("Final music saved as 'final_music.wav'")
        
    # except Exception as e:
    #     print(f"Error saving music: {str(e)}")
    #     print("Try playing the piano_sequence.wav instead")


    try:
        final_audio, sample_rate = generate_final_music(piano_audio, music_prompt)

        # Ensure the audio data is in the correct range and format
        final_audio = np.clip(final_audio, -32768, 32767).astype(np.int16)

        # Save as WAV file
        scipy.io.wavfile.write("final_music.wav", sample_rate, final_audio)

        print("Final music saved as 'final_music.wav'")

    except Exception as e:
        print(f"Error saving music: {str(e)}")
        print("Try playing the piano_sequence.wav instead")        
        
if __name__ == "__main__":
    main()