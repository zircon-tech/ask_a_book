import pinecone
import os
from utils import create_embedding, get_text_from_file
import openai
import argparse
from dotenv import load_dotenv
from textwrap import wrap

load_dotenv()


pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def search(query, index_name="document-chunks"):
    embedding = create_embedding(query)
    pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")
    index = pinecone.Index(index_name)
    results = index.query(queries=[embedding], top_k=1)
    nearest_chunk_id = results["results"][0]["matches"][0]["id"]
    return nearest_chunk_id


def gpt3_completion(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message


def main():
    parser = argparse.ArgumentParser(
        description='Read text from different file formats')
    parser.add_argument('file', help='Path to the file to read text from')
    args = parser.parse_args()

    input_folder = "working"
    file_path = os.path.join(input_folder, args.file)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Prompt user for a question
    question = input("Enter your question: ")

    index_name = f"document-chunks-{file_name}"

    # Search for the nearest chunk and get its ID
    nearest_chunk_id = search(question, index_name)

    # Get the chunk index from the ID
    chunk_index = int(nearest_chunk_id.split("-")[-1])

    text = get_text_from_file(file_path)

    chunks = wrap(text, 2000)  # TODO: Store the chunks in a file

    # Retrieve the corresponding text chunk from the 'chunks' list
    nearest_chunk_text = chunks[chunk_index]

    # Use GPT-3 to answer the question based on the retrieved chunk
    prompt = f"The following text contains the information you are looking for:\n{nearest_chunk_text}\n\nQuestion: {question}\nAnswer:"
    answer = gpt3_completion(prompt)

    # Print the answer
    print("Answer:", answer)


if __name__ == "__main__":
    main()
