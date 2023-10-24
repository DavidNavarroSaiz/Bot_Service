

<h3 align="center">Support chatbot service</h3>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project leverages neural networks and word embedding techniques to build an intelligent chatbot. 
the neural network model aims to respond simple parts of the conversations just like grettings, thanks and goodbyes, and word embedding is implemented to answer techical and more complex questions, to answer those questios a vectors database is created from the initial pdf, then based on the input of the user, it search similarity and extracts the 3 more similar paragraphs of the pdf, then that text is processed using ChatGPT and the answer is returned.

The bot is designed to understand and respond to user queries and conversations effectively. Some key aspects of the project include:

- **Neural Networks**: The project employs deep learning techniques to create a neural network model that can understand and generate human-like responses.

- **Word Embedding**: Word embedding methods are utilized to convert words into numerical vectors. This enables the model to process and comprehend text data efficiently.

- **Knowledge Base**: The project includes a knowledge base built from the content of the "Deep Learning.pdf" document. The bot can utilize this knowledge base to provide informative responses.

- **FastAPI Service**: The FastAPI service, implemented in "main.py," acts as an interface for users to interact with the chatbot. It provides RESTful APIs for sending and receiving messages.

This project is aimed at demonstrating the potential of natural language processing and neural networks for creating intelligent conversational agents.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This project makes use of various technologies and libraries to achieve its goals. Some of the key components include:

- **Python**: The project is primarily written in Python, making use of its extensive libraries for natural language processing and machine learning.

- **FastAPI**: The FastAPI framework is used to create a high-performance RESTful API for the chatbot, enabling efficient communication with users.

- **PyTorch**: PyTorch is employed to build and train the neural network models, allowing for deep learning-based natural language understanding and generation.

- **NLTK (Natural Language Toolkit)**: NLTK is used for text preprocessing, tokenization, and other NLP-related tasks.

- **Word Embeddings**: Techniques like Word2Vec, GloVe, or other word embedding methods are utilized to convert text data into numerical vectors for model input.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



Follow these steps to get started with the project on your local machine.

### Prerequisites

Make sure you have the following prerequisites installed:

- Python (3.7 or higher)
- Git (for version control)

### Installation

1. Clone the repository to your local machine:

    ```
    git clone https://github.com/DavidNavarroSaiz/Bot_Service
    ```

2. Navigate to the project directory:

    ``` 
    cd your-project
    ```

3. Install project dependencies using `pip`:

    ```
    pip install -r requirements.txt
    ```

### Training the Neural Network

To train the neural network model, follow these steps:


1. Run the training script:

    ```
    python train.py
    ```
    

The script will train the model using the provided data in the `intents.json` file and will create a file with the model, called `data.pth`

### Using the FastAPI Service

To use the FastAPI service for interacting with the chatbot, follow these steps:

1. Run the FastAPI service:

    ```
    python main.py
    ```
    or using the console:

    ```
    uvicorn main:app --reload --port 8000
    ```

2. The service will start on a specified port (typically 8000). You can access the API endpoints using tools like `curl` or API testing platforms.

### Usage

Provide examples and instructions on how to use the FastAPI service to interact with the chatbot. Include details on endpoints, requests, and responses.

```
curl -X POST http://localhost:8000/ask_question -d "message=Hello, chatbot!"

```

to format code using black:

```
black ./Bot_Service
```
To pylint and evaluate your code:
```
pylint ./Bot_Service
```