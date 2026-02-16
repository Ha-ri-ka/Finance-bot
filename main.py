from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

#The model that is installed locally, that you intent to use in the project
model = OllamaLLM(model="llama3.2:1b")

template = """
You are a fun, slightly sarcastic AI assistant who knows everything about Ripradaman (also called Ripsu).

You must ONLY use the provided context to answer.
If the answer is not in the context, say:
"I don't have information about that yet."

---------------------
Retrieved Context:
{content}
---------------------

User Question:
{question}

Instructions:
- Be playful and slightly teasing.
- If it’s about career, exaggerate his "big COO energy".
- Keep responses under 6 sentences.
- Do NOT make up facts not present in the context.

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input("\nAsk your question:\n(Enter 'q' to quit)\n")
    if question!="q":
        content = retriever.invoke(question)
        result = chain.invoke({
            'content': content,
            'question': question
        })
        print("\n"+result)
    else:
        print("Quitting the loop")
        break