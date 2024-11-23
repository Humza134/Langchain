import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Streamlit framework
st.title("Celebrity Search")
input_text = st.text_input("Search the topic you want")

# Prompt templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Memory instances
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
desc_memory = ConversationBufferMemory(input_key='dob', memory_key='desc_history')

# Chains for the LLM
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=desc_memory)

# Combine chains into a SimpleSequentialChain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# Streamlit display logic
if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(desc_memory.buffer)
