from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize OpenAI embedding model
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def get_text_embedding(text):
    """Generate embedding for a given text using OpenAI API."""
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def select_best_resume(job_description, resumes):
    """
    Select the most relevant resume based on similarity score.
    Uses OpenAI embeddings and cosine similarity.
    """
    job_embedding = get_text_embedding(job_description)

    best_match = None
    highest_score = -1

    for filename, data in resumes.items():
        resume_text = data['text']  # ✅ Extract text, not file path
        resume_embedding = get_text_embedding(resume_text)
        similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

        if similarity > highest_score:
            highest_score = similarity
            best_match = filename

    return best_match