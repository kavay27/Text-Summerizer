import os
import pickle
import torch
import asyncio
from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import time

# Ensure an event loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the summarizer with a specific BART model
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded successfully. Using device: {self.device}")
    
    def summarize(self, text, max_length=150, min_length=40, length_penalty=2.0, 
                  num_beams=4, early_stopping=True):
        """
        Summarize the given text
        """
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(self.device)
        summary_ids = self.model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=length_penalty,
            num_beams=num_beams, 
            early_stopping=early_stopping
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Load the model without pickle
summarizer = TextSummarizer()

# Streamlit UI
st.title("BART Text Summarizer")
st.write("Enter text below to generate a summary using the BART model.")

text = st.text_area("Input Text", "", height=200)
max_length = st.number_input("Max Summary Length", min_value=10, max_value=500, value=150)
min_length = st.number_input("Min Summary Length", min_value=10, max_value=500, value=40)

if st.button("Summarize"):
    if text.strip():
        summary = summarizer.summarize(text, max_length=max_length, min_length=min_length)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
