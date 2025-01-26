from langchain.prompts import PromptTemplate  # Import PromptTemplate class from langchain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import re  # Import the 're' module for regular expressions
import streamlit as st  # Import Streamlit for web app development

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Intialize the Gemini model
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define a PromptTemplate for title suggestions
prompt_template_for_title_suggestion = PromptTemplate(
    input_variables=['topic'],  # Specify input variables
    template =  # Define the prompt template
    '''
    I'm planning a blog post on topic : {topic}.
    The title is informative, or humorous, or persuasive. 
    The target audience is beginners, tech enthusiasts.  
    Suggest a list of ten creative and attention-grabbing titles for this blog post. 
    Don't give any explanation or overview to each title.
    '''
)

title_suggestion_chain = prompt_template_for_title_suggestion | llm # defining the title suggestion chain

# Define a PromptTemplate for blog content generation
prompt_template_for_title = PromptTemplate(
    input_variables=['title', 'keywords', 'blog_length'],  # Specify input variables
    template=  # Define the prompt template
    '''Write a high-quality, informative, and plagiarism-free blog post on the topic: "{title}". 
    Target the content towards a beginner audience. 
    Use a conversational writing style and structure the content with an introduction, body paragraphs, and a conclusion. 
    Try to incorporate these keywords: {keywords}. 
    Aim for a content length of {blog_length} words. 
    Make the content engaging and capture the reader's attention.'''
)

title_chain = prompt_template_for_title | llm # Create a chain for title generation

## Working on UI with the help of streamlit
st.title("AI Blog Content Assistant...🤖")
st.header("Create High-Quality Blog Content Without Breaking the Bank")

st.subheader('Title Generation') # Display a subheader for the title generation section
topic_expander = st.expander("Input the topic") # Create an expander for topic input

# Create a content block within the topic expander
with topic_expander:
    topic_name = st.text_input("Enter the Topic", key="topic_name") # Get user input for the topic name
    submit_topic = st.button('Submit topic') # Button for submitting the topic

if submit_topic:  # Handle button click (submit_topic)
    title_selection_text = ''  # Initialize an empty string to store title suggestions
    title_suggestion = title_suggestion_chain.invoke({"topic": topic_name})  # Pass the topic properly as a dict
    if hasattr(title_suggestion, "content"):  # Check if the returned object has 'content'
        title_suggestion_str = title_suggestion.content  # Extract the text content
        for sentence in title_suggestion_str.split('\n'): 
            title_selection_text += (sentence.strip() + '\n')  # Clean up each sentence and add it to the selection text
        st.text(title_selection_text)  # Display the generated title suggestions
    else:
        st.error("Failed to generate titles. Please try again.")


st.subheader('Blog Generation') # Display a subheader for the blog generation section
title_expander = st.expander("Input the title") # Create an expander for title input


with title_expander: # Create a content block within the title expander
    title_of_the_blog = st.text_input("Enter the title", key="title_of_the_blog") # Get user input for the blog title
    num_of_words = st.slider('Number of Words', min_value=100, max_value=1000, step=50) # Slider for selecting the desired number of words


    if 'keywords' not in st.session_state: # Manage keyword list in session state
        st.session_state['keywords'] = []  # Initialize empty list on first run
    keyword_input = st.text_input("Enter a keyword:") # Input field for adding keywords
    keyword_button = st.button("Add Keyword") # Button to add keyword to the list
    if keyword_button: # Handle button click for adding keyword
        st.session_state['keywords'].append(keyword_input) # Add the keyword to the session state list
        st.session_state['keyword_input'] = "" # Clear the keyword input field after adding
        for keyword in st.session_state['keywords']:  # Display the current list of keywords
            # Inline styling for displaying keywords
            st.write(f"<div style='display: inline-block; background-color: lightgray; padding: 5px; margin: 5px;'>{keyword}</div>", unsafe_allow_html=True)

    # Button to submit the information for content generation
    submit_title = st.button('Submit Info')

if submit_title: # Handle button click for submitting information
    formatted_keywords = []
    for i in st.session_state['keywords']: # Process and format keywords
        if len(i) > 0:
            formatted_keywords.append(i.lstrip('0123456789 : ').strip('"').strip("'"))  
    formatted_keywords = ', '.join(formatted_keywords)

    st.subheader(title_of_the_blog) # Display the blog title as a subheader
    st.write(title_chain.invoke({'title': title_of_the_blog, 'keywords': formatted_keywords, 'blog_length':num_of_words})) # Generate and display the blog content using the title chain

