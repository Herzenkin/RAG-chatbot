RAG Chatbot for "Databases for GenAI" lecture.

This script contains next functionality:
1.  Loading Data:
    - Converts PDF pages to images and uses a Gemini VLM API to extract text.
    - Transcribes audio from video file using AWS Whisper.
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
    - Add this key into the `GEMINI_API_KEY` variable.
4.  Set next properties for data files:
    - `PDF_PATH` is the path to the PDF file.
    - `VIDEO_PATH` is the path to the video file.
5.  Run the script:
    `python rag_pipeline.py`