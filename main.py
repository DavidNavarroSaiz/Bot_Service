import random
import json
import torch
import pickle
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from nltk_utils import bag_of_words, tokenize
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import NeuralNet
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()
# Add CORS middleware for handling cross-origin requests

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def process_text(text):
    """
    Processes text by splitting it into chunks and converting them to embeddings.

    Args:
        text: The text to be processed.

    Returns:
        A FAISS object containing the text embeddings.
    """
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    with open("knowledgeBase.pkl", "wb") as pickle_file:
        pickle.dump(knowledgeBase, pickle_file)
    return knowledgeBase

try:
    with open("knowledgeBase.pkl", "rb") as pickle_file:
        knowledgeBase = pickle.load(pickle_file)
        print("pickle loaded")
except FileNotFoundError:
    pdf_path = './../Deep Learning.pdf'
    pdf_reader = PdfReader(pdf_path)
    # Text variable will store the pdf text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Create the knowledge base object
    knowledgeBase = process_text(text)
    print("pdf loaded")

class QuestionInput(BaseModel):
    query: str



# Define your routes using FastAPI decorators
@app.get("/")
def index_get():
    return {"message": "Welcome to the FastAPI app"}
# def filter_results_by_score(results, threshold):
#     return [(doc, score) for doc, score in results if score >= threshold]
def filter_results_by_score(results, threshold):
  """Returns a list of (doc, score) tuples for each case where the score is greater than or equal to the threshold.

  Args:
    results: A list of (doc, score) tuples, where doc is the document and score is the relevance score.
    threshold: The threshold score.

  Returns:
    A list of (doc, score) tuples for each case where the score is greater than or equal to the threshold.
  """

  filtered_results = []
  for doc, score in results:
    # print("doc",doc)
    # print("score",score)
    if score >= threshold:
      filtered_results.append((doc))
  return filtered_results

# Use the function
@app.post("/ask_question")
async def ask_question(question_input: QuestionInput):
    """Answers a question using the trained neural network model and the knowledge base.

    Args:
        question_input: A Pydantic model containing the question to be answered.

    Returns:
        A JSON object containing the answer to the question.
    """
    query = question_input.query
    response = "please enter a valid question"  # Default response if query is not provided or an error occurs
    
    if query:
        sentence = tokenize(query)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        # print("prob",prob)
        if prob.item() > 0.95:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    print("NN response",response)
                    return {"answer": response}
        
        else:
            docs = knowledgeBase.similarity_search_with_relevance_scores(query,4)
            docs = filter_results_by_score(docs, 0.7)  # 
            if len(docs) != 0:
                llm = OpenAI(temperature=0.1)
                chain = load_qa_chain(llm, chain_type='stuff')
                
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                print("Embedding response",response)
                return {"answer": response}
            else:
                print("no query",response)
                return {"answer": response}
    else:
        print("no query",response)
        return {"answer": response}


# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
