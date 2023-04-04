# py-langchain-bot

A chatbot that is using chat-gpt and custom training-data from pdf files. 

The easiest way to test it yourself is to use the google cloab platform. There you can write and run a python code without any installation effort.

1. If you don't have a google-account jet create one
2. Login to https://colab.research.google.com
3. Create a "colab notebook"
4. Create a "Code" entry and run it. This code installs the required packages and defines the 2 functions `construct_index()` and `ask_bot()`.
```python
!pip install llama-index
!pip install langchain
!pip install PyPDF2

#from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
import sys
#from google.colab import drive
import os
import PyPDF2

def construct_index(directory_path):
  # set maximum input size
  max_input_size = 4096
  # set number of output tokens
  num_outputs = 256
  # set maximum chunk overlap
  max_chunk_overlap = 20
  # set chunk size limit
  chunk_size_limit = 600

  prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs))
  
  documents = SimpleDirectoryReader(directory_path).load_data()
  
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
  index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
  
  index.save_to_disk('index.json')
  
  return index

def ask_bot(input_index = 'index.json'):
  index = GPTSimpleVectorIndex.load_from_disk(input_index)
  while True:
    query = input('What do you want to ask the bot?   \n')
    response = index.query(query, response_mode="compact")
    print ("\nBot says: \n\n" + response.response + "\n\n\n")
```

5. Now we can load the custom data we want to train the chat with. The files must be accessible from the internet. Set the url accordingly. As an example you could enter and execute following code:
```python
!wget https://yourhost/chat-gpt/Portfolio-Spick.pdf
!wget https://yourhost/chat-gpt/Wieso-wir.pdf
!wget https://yourhost/chat-gpt/Was-wir-bieten.pdf
!wget https://yourhost/chat-gpt/Technologien.pdf
!wget https://yourhost/chat-gpt/uuebkit.pdf
```
6. Set the API-Key you got from OPEN-AI, by running this code
```python
os.environ["OPENAI_API_KEY"] = 'your-open-ai-api-key'
```
7. Create the index of your custom data files, by running this code
```python
index = construct_index("/content/")
```
8. Now you can start the chat-bot
```python
ask_bot('index.json')
```

