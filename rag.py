import json
import os
import argparse

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# Settings
CV_FILE = "cv_data.json"
INDEX_FILE = "cv_index.faiss"
CHUNKS_FILE = "cv_chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

TOP_K = 4


# Load CV
def load_cv():
    with open(CV_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Convert CV into simple text chunks
def create_chunks(cv):
    chunks = []

    p = cv["personal_details"]
    chunks.append(
        f"Personal Details: {p['name']}. "
        f"Location: {p['location']}. "
        f"Summary: {p['summary']}"
    )

    for edu in cv.get("education", []):
        chunks.append(
            f"Education: {edu['degree']} in {edu['field']} "
            f"at {edu['institution']}. "
            f"CGPA: {edu['cgpa']}."
        )

    for category, skills in cv.get("skills", {}).items():
        chunks.append(
            f"Skills - {category}: {', '.join(skills)}"
        )

    for project in cv.get("projects", []):
        chunks.append(
            f"Project: {project['name']}. "
            f"Role: {project['role']}. "
            f"Technologies: {', '.join(project['technologies'])}. "
            f"Description: {project['description']}"
        )

    for exp in cv.get("experience", []):
        chunks.append(
            f"Experience: {exp['title']} at {exp['company']}. "
            f"Duration: {exp['start_date']} - {exp['end_date']}. "
            f"Achievements: {', '.join(exp.get('achievements', []))}"
        )

    return chunks


# Build FAISS index
def build_index(chunks, model):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Index created successfully.")


# Load FAISS index
def load_index():
    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


# Search relevant CV information
def search(question, model, index, chunks):
    embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embedding)

    _, results = index.search(
        embedding,
        TOP_K
    )

    return [
        chunks[i]
        for i in results[0]
        if i >= 0
    ]


# Ask Gemini
def ask_gemini(question, context):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Please set GEMINI_API_KEY in your .env file."

    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = f"""
Answer the question using ONLY the CV information below.
Do not make up information.

CV Information:
{chr(10).join(context)}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text



# Complete RAG process
def ask(question, model, index, chunks):
    context = search(
        question,
        model,
        index,
        chunks
    )

    return ask_gemini(
        question,
        context
    )


# Main program
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query")

    args = parser.parse_args()

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Build index
    if args.build:
        cv = load_cv()
        chunks = create_chunks(cv)
        build_index(chunks, model)
        return

    # Build automatically if index doesn't exist
    if not os.path.exists(INDEX_FILE):
        cv = load_cv()
        chunks = create_chunks(cv)
        build_index(chunks, model)

    index, chunks = load_index()

    # Single query
    if args.query:
        print(
            ask(
                args.query,
                model,
                index,
                chunks
            )
        )
        return

    # Chat
    print("\nCV Chatbot")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ")

        if question.lower() in ["quit", "exit"]:
            break

        answer = ask(
            question,
            model,
            index,
            chunks
        )

        print("Bot:", answer)


if __name__ == "__main__":
    main()
