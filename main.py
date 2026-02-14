from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

#The model that is installed locally, that you intent to use in the project
model = OllamaLLM(model="llama3.2:1b")


