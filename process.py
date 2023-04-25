from utils import create_embedding, get_text_from_file
import numpy as np
import openai
import pinecone
from textwrap import wrap
import pdfplumber
from docx import Document
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.environ.get("OPENAI_API_KEY")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")


def main():
    parser = argparse.ArgumentParser(
        description='Read text from different file formats')
    parser.add_argument('file', help='Path to the file to read text from')
    args = parser.parse_args()

    input_folder = "working"
    file_path = os.path.join(input_folder, args.file)
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    text = get_text_from_file(file_path, file_ext)

    chunks = wrap(text, 2000)  # TODO: Store the chunks in a file

    embeddings = []
    for chunk in chunks:
        embedding = create_embedding(chunk)
        embeddings.append(embedding)

    index_name = f"document-chunks-{file_name}"

    pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")
    if index_name not in pinecone.list_indexes():
        embeddings = np.array(embeddings)
        dimension = embeddings[0].shape[0]
        pinecone.create_index(index_name, metric="cosine", dimension=dimension)

    index = pinecone.Index(index_name)

    upserts = [(f"chunk-{i}", embedding)
               for i, embedding in enumerate(embeddings)]
    index.upsert(vectors=upserts)

    print(
        f"Successfully indexed {len(chunks)} chunks of text. You can now use app.py")


if __name__ == "__main__":
    main()
