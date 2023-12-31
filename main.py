"""_summary_
Chatbot service implemented in FastAPI use neural network and 
word embedding to have a fluent conversation with the user. 

Returns:
    _type_: _description_
"""
import json
import os
import pickle
import random

import torch
from PyPDF2 import PdfReader

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import openai

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

# Add CORS middleware for handling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents from JSON file
with open("./docs/intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Load data from a pre-trained model
FILE = "./docs/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize the neural network model
model_nn = NeuralNet(input_size, hidden_size, output_size).to(device)
model_nn.load_state_dict(model_state)
model_nn.eval()

# Define a function to process text and create a knowledge base
def process_text(text_process):
    """
    Processes text by splitting it into chunks and converting them to embeddings.

    Args:
        text_process: The text to be processed.

    Returns:
        A FAISS object containing the text embeddings.
    """
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text_process)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    with open("./docs/knowledgeBase.pkl", "wb") as pickle_f:
        pickle.dump(knowledge_base, pickle_f)
    
    return knowledge_base

# Load or create knowledge base
try:
    with open("./docs/knowledgeBase.pkl", "rb") as pickle_file:
        knowledgeBase_pkl = pickle.load(pickle_file)
        print("pickle loaded")
except FileNotFoundError:
    pdf_path = "./docs/Deep Learning.pdf"
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    knowledgeBase_pkl = process_text(text)
    print("pdf loaded")

# Define routes using FastAPI decorators
@app.get("/")
def index_get():
    """
    Responds to a GET request at the root URL.
    Returns a welcome message.
    """
    return {"message": "Welcome to the FastAPI app"}

# Define a function to filter results by relevance score
def filter_results_by_score(results, threshold):
    """
    Filters a list of results by a relevance threshold.

    Args:
        results (list): A list of (doc, score) pairs.
        threshold (float): The minimum score required to keep a result.

    Returns:
        list: A filtered list of (doc, score) pairs.
    """
    return [doc for doc, score in results if score >= threshold]

# Pydantic model for question input
class QuestionInput(BaseModel):
    """
    Pydantic model for incoming question input.
    """
    query: str

# Use the function
@app.post("/ask_question")
async def ask_question(question_input: QuestionInput):
    """
    Responds to a POST request at the "/ask_question" endpoint.
    Processes the input question and returns an answer.

    Args:
        question_input (QuestionInput): Input question data.

    Returns:
        dict: A response containing an answer or an error message.
    """
    query = question_input.query
    response = "Please enter a valid question"  # Default response if query is not provided or an error occurs

    if query:
        sentence = tokenize(query)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model_nn(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        print("prob: ",prob)
        # TODO: parametro
        if prob.item() > 0.998:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    response = random.choice(intent["responses"])
                    print("NN response", response)
                    return {"answer": response}
        else:
            docs_scores = knowledgeBase_pkl.similarity_search_with_relevance_scores(query, 4)
            docs = filter_results_by_score(docs_scores, 0.7)
            
            if len(docs) != 0:
                llm = OpenAI(temperature=0.1)
                chain = load_qa_chain(llm, chain_type="stuff")

                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                print("Embedding response", response)
                return {"answer": response}
            else:
                print("no query", response)
                return {"answer": response}
    else:
        print("no query", response)
        return {"answer": response}

# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
