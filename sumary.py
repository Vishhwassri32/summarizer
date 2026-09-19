import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BartForConditionalGeneration, BartTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
import torch
from transformers import AutoTokenizer


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import torch
from transformers import BartTokenizer, BartForConditionalGeneration


model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move to GPU if available
device = "cpu"
model = model.to(device)

def get_text_input(user_input):
                """
                Checks if the input is a valid file path. 
                If yes, it reads the file. If no, it treats the input as direct text.
                """
                # Check if user passed a file path that exists on the system
                if isinstance(user_input, str) and os.path.exists(user_input):
                        print(f"File detected: Loading {user_input}...")
                        if user_input.endswith('.txt'):
                                loader = TextLoader(user_input)
                        elif user_input.endswith('.pdf'):
                                loader = PyPDFLoader(user_input)
                        elif user_input.endswith('.docx'):
                                loader = Docx2txtLoader(user_input)
                        else:
                         raise ValueError("Unsupported file format. Use .txt, .pdf, or .docx")
                        
                        docs = loader.load()
                        return " ".join([doc.page_content for doc in docs])
                
                # If it's not an existing file path, assume it's raw paragraph text
                else:
                        print("Raw text paragraph detected...")
                        return user_input



def chunk_text(text, chunk_size=1000, overlap=200):
                """
                Splits the text into smaller chunks for processing.
                """
                text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        length_function=len,
                        separators=["\n\n", "\n", ".", ""]
                )
                return text_splitter.split_text(text)


def summarize_text(chunks):
                """
                Summarizes each chunk of text using the BART model.
                """
                summaries = []
                for chunk in chunks:
                        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
                        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
                        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        summaries.append(summary)
                return " ".join(summaries)

if __name__ == "__main__":
    user_input = input("Enter a file path or a paragraph of text: ")
    text_content = get_text_input(user_input)
    chunks = chunk_text(text_content)
    summaries = summarize_text(chunks)
    print("Extracted Text Content:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
    print("\nSummarized Content:")
    print(summaries)