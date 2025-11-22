"""
RAG Chatbot for "Databases for GenAI" lecture.

This script contains next functionality:
1.  Loading Data:
    - Converts PDF pages to images and uses a Gemini VLM API to extract text.
    - Transcribes audio from video file using OpenAI Whisper.
2.  Chunking Text:
    - Splits the combined text into semantically meaningful chunks.
3.  Embedding & Storing Data:
    - Converts chunks into vector embeddings and stores them in ChromaDB.
4.  Retrieving & Generating Answer:
    - Takes a user question and embeds it.
    - Retrieves relevant context chunks from ChromaDB.
    - Passes the question and context to the Gemini LLM to generate a final answer.

To run this pipeline::
1.  Install Poppler, a critical dependency for PDF-to-image conversion.
    - On macOS: `brew install poppler`
    - On Linux: `sudo apt-get install poppler-utils`
2.  Install all required Python dependencies::
    `pip install -r requirements.txt`
3.  Get a Gemini API Key:
    - Create an API key via Google AI Studio (https://aistudio.google.com/app/apikey)
    - Add this key into the `GEMINI_API_KEY` variable below.
4.  Set next properties for data files:
    - `PDF_PATH` is the path to the PDF file.
    - `VIDEO_PATH` is the path to the video file.
5.  Run the script:
    `python rag_pipeline.py`
"""

import base64
import io
import json
import os
import time

import chromadb
import librosa
import requests
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from moviepy.video.io.VideoFileClip import VideoFileClip
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- CONFIGURATION ---

# Data files
PDF_PATH = "Databases_for_GenAI.pdf"
VIDEO_PATH = "Databases_for_GenAI.mp4"

# ChromaDB
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "genai_datastore"

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
WHISPER_MODEL_NAME = "openai/whisper-base"
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# IMPORTANT: SET YOUR GEMINI API KEY HERE
GEMINI_API_KEY = ""

# Parameters to tweak
CHUNK_SIZE = 1000 # Number of characters used for text chunking
CHUNK_OVERLAP = 100 # Number of characters for overlapping text between chunks
CONTEXT_CHUNK_NUMBER = 10 # Number of results used for RAG context when generating answer


def extract_text_from_images(image_list: list) -> str:
    """
    Extracts text content from a list of PDF page images using the Gemini VLM API.
    This function iterates through the provided images, converts them to base64,
    and sends them to the Gemini API for text transcription. It applies
    exponential backoff for API robustness and includes a delay to manage
    rate limits (15 QPM for the free tier).

    Args:
        image_list (list): a list of Image objects, where each image represents a page from the source PDF.

    Returns:
        str: a string with all the extracted text from all pages.
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is not set.")
        return ""

    print(f"Extracting text from {len(image_list)} PDF pages using Gemini...")

    full_text = ""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}

    # Process one image at a time to keep requests simple
    for i, img in enumerate(image_list):
        print(f"  - Processing page {i + 1}/{len(image_list)}...")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Extract all text visible in this image of a presentation slide. Preserve formatting like newlines. Output only the transcribed text and nothing else."},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ],
            # force Gemini to be as uncreative and literal as possible, just text transcription only
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.95,
                "topK": 40,
            }
        }

        try:
            # exponential backoff for API calls
            max_retries = 3
            delay = 1
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()

                    if "candidates" in result and result["candidates"]:
                        print(f"  - {result["candidates"]}")

                        page_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        full_text += page_text + "\n\n"

                        print(f"  - Page {i + 1}/{len(image_list)} processed successfully.")
                        break
                    else:
                        print(f"    - Page {i + 1} Gemini extraction failed or returned no content. Retrying...")

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    print(f"    - API Error: {e}. Retrying (Attempt {attempt + 2}/{max_retries})...")

        except requests.exceptions.RequestException as e:
            print(f"    - Page {i + 1} API Error: {e}")
            if e.response:
                print(f"    - Response body: {e.response.text}")

        # Free tiers of Gemini API has limit for requests per minute (15 QPM), so we "delay" the next request to fit it
        if i < len(image_list) - 1:  # Don't sleep after the last page
            time.sleep(4)

    print(f"Successfully extracted {len(full_text)} characters from PDF images.")
    return full_text


def load_pdf_and_extract_text(file_path: str) -> str:
    """
    Converts a PDF file into a sequence of images and delegates text extraction
    to the Vision Language Model (VLM) via `extract_text_from_images`.

    Args:
        file_path (str): the local path to the input PDF file.

    Returns:
        str: the combined extracted text from all PDF pages.
    """
    print(f"Converting PDF {file_path} to images for Gemini processing...")

    try:
        images = convert_from_path(file_path)

        if not images:
            print("Error: PDF conversion returned no images.")
            return ""

        return extract_text_from_images(images)

    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return ""


def extract_audio_from_video(video_path: str) -> str:
    """
    Extracts the audio track from a video file and saves it as a temporary WAV file.
    This temporary file is used for subsequent transcription.

    Args:
        video_path (str): the local path to the input video file.

    Returns:
        str: the file path of the temporary WAV audio file.
    """
    print(f"Extracting audio from {video_path}...")

    audio_path = "temp_audio.wav"
    try:
        with VideoFileClip(video_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path, fps=16000, codec='pcm_s16le')

        print(f"Successfully extracted audio to {audio_path}")
        return audio_path
    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}")
        return ""
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return ""


def transcribe_audio(audio_path: str, whisper_pipeline) -> str:
    """
    Transcribes the speech content from an audio file using OpenAI Whisper model.
    The temporary audio file is deleted afterward.

    Args:
        audio_path (str): the local path to the temporary WAV audio file.
        whisper_pipeline: A loaded Whisper `pipeline` object for automatic speech recognition.

    Returns:
        str: the full transcribed text from the audio file.
    """
    print(f"Transcribing audio from {audio_path}...")

    try:
        # Transform audio file into the correct format for OpenAI Whisper
        speech, sample_rate = librosa.load(audio_path, sr=16000)

        result = whisper_pipeline(
            speech,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={"language": "en"}
        )
        transcription = result["text"]

        print(f"Here is the transcribed text:\n\n{transcription}")
        print(f"Successfully transcribed {len(transcription)} characters from audio.")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)


def chunk_text(full_text: str) -> list[str]:
    """
    Splits the combined document and transcription text into smaller, semantically meaningful chunks for embedding.
    Uses LangChain's RecursiveCharacterTextSplitter for robust chunking that attempts
    to keep content coherent based on separators.

    Args:
        full_text (str): the entire combined text from all sources (PDF and video).

    Returns:
        list[str]: a list of text strings (chunks).
    """
    print(f"Chunking text of {len(full_text)} characters...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    docs = text_splitter.create_documents([full_text])
    chunks = [doc.page_content for doc in docs]

    print(f"Split text into {len(chunks)} chunks.")
    return chunks


def setup_vector_storage(chunks: list[str], embedding_model) -> chromadb.Collection:
    """
    Initializes a ChromaDB instance, generates vector embeddings for all text chunks,
    and stores the chunks and embeddings in a collection.

    Args:
        chunks (list[str]): the list of text chunks to be stored.
        embedding_model: a loaded SentenceTransformer model used to generate the vector embeddings.

    Returns:
        chromadb.Collection: The ChromaDB collection object containing the stored data.
    """
    print("Setting up ChromaDB storage...")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Generate IDs for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"Generating {len(chunks)} embeddings... (This may take a moment)")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()

    print("Adding documents to ChromaDB collection...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_documents = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings
        )

    print(f"Successfully added {collection.count()} documents to collection '{COLLECTION_NAME}'.")
    return collection


def retrieve_context(question: str, collection: chromadb.Collection, embedding_model) -> str:
    """
    Retrieves the most relevant text chunks from the vector database based on the user's question.
    The question is first embedded, and then a similarity search is performed against the stored chunk embeddings.

    Args:
        question (str): the user's input question.
        collection (chromadb.Collection): the ChromaDB collection to query.
        embedding_model: the SentenceTransformer model used for embedding.

    Returns:
        str: a string concatenating the retrieved documents, separated by delimiters for the LLM.
    """
    print(f"Retrieving context for question: '{question}'")

    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=CONTEXT_CHUNK_NUMBER,
        include=["documents"]
    )

    context = "\n\n---\n\n".join(results['documents'][0])
    print(f"Retrieved context for question: '{context}'")
    return context


def generate_answer(question: str, context: str) -> str:
    """
    Generates a final answer to the user's question using the Gemini API.
    The model is instructed via a system prompt to strictly use the provided
    context and to refuse to answer if the information is unavailable.

    Args:
        question (str): the user's original question.
        context (str): the relevant text retrieved from the vector store.

    Returns:
        str: the final generated answer from the LLM.
    """
    print("Generating answer with Gemini...")

    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set."

    # This is the core prompt for RAG.
    # It instructs the model to answer based only on the provided context.
    system_prompt = f"""
    You are a helpful assistant for the "Databases for GenAI" course.
    Your task is to answer the user's question based only on the provided context.
    The context contains transcribed text from the lecture and text from the presentation slides.
    If the context does not contain the answer, state clearly that you cannot answer the question based on the provided information.
    Do not make up information or use any external knowledge.

    Here is the context:
    ---
    {context}
    ---
    """

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    headers = {'Content-Type': 'application/json'}

    payload = {
        "contents": [
            {
                "parts": [{"text": question}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.5,
            "topP": 0.95,
            "topK": 40,
        }
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        if "candidates" in result and result["candidates"]:
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer
        else:
            return "Error: No content generated."

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        if e.response:
            print(f"Response body: {e.response.text}")
        return f"Error: Failed to connect to Gemini API. {e}"


def main():
    """
    The main execution function for the RAG chatbot pipeline.
    It performs the following steps:
    1. Loads the embedding and transcription models.
    2. Extracts text from the PDF slides (Gemini VLM) and transcribes the video audio (OpenAI Whisper).
    3. Chunks the combined text.
    4. Embeds the chunks and stores them in ChromaDB.
    5. Enters a loop to continuously take user input, retrieve context, and generate answers.
    """
    print("--- Starting RAG Chatbot Pipeline ---")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    print(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        device=device
    )
    print("Models loaded.")

    # --- DATA LOADING PHASE ---
    # Step 1: Load Data
    pdf_text = load_pdf_and_extract_text(PDF_PATH)
    audio_file_path = extract_audio_from_video(VIDEO_PATH)

    audio_text = ""
    if audio_file_path:
        audio_text = transcribe_audio(audio_file_path, whisper_pipeline)

    if not pdf_text and not audio_text:
        print("Error: No text could be extracted from any source. Exiting.")
        return

    full_text = pdf_text + "\n\n--- LECTURE TRANSCRIPT ---\n\n" + audio_text

    # Step 2: Chunk Text
    chunks = chunk_text(full_text)

    if not chunks:
        print("Error: No chunks were created. Exiting.")
        return

    # Step 3: Embed and Store
    try:
        vector_collection = setup_vector_storage(chunks, embedding_model)
        print("--- Data loading complete: Vector Store is Ready ---")
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return

    # --- QUERY PHASE ---
    print("\nStarting chatbot. Type 'quit' to exit.")

    while True:
        question = input("\nAsk your question: ")
        if question.lower() == 'quit':
            break

        # Step 4: Retrieve & Generate
        context = retrieve_context(question, vector_collection, embedding_model)

        if not context:
            print("No relevant context found for your question.")
            continue

        answer = generate_answer(question, context)

        print("\n--- Answer ---")
        print(answer)
        print("--------------")


if __name__ == "__main__":
    main()