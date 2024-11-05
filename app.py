from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# system_message = [SystemMessage(content="You are a python programmer assistant")]
# human_message = [HumanMessage(content="{question}")]
# chat_prompt = ChatPromptTemplate.from_messages(system_message + human_message)
# output_parser = StrOutputParser()
# chain = chat_prompt | llm | output_parser

# result = chain.invoke({"question": "write a function to sort a list in python"})

# print(result)

messages = [
    SystemMessage(content="Translate the following into {language}:"),
    HumanMessage(content="{text}")
]

prompt = ChatPromptTemplate.from_messages(messages)

parser = StrOutputParser()

chain = prompt | llm | parser

chain.invoke({"language": "French", "text": "I love programming"})