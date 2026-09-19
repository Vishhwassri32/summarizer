import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# Load environment variables from the .env file
load_dotenv()

# Client automatically looks for the GEMINI_API_KEY environment variable
client = genai.Client()

def main_function(text_content: str) -> str:
    prompt = f"Summarize the following text comprehensively:\n\n{text_content}"
    
    try:
        # First attempt with the standard flash model
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text if response.text else "[Error: Gemini returned empty response]"
        
    except Exception as e:
        # Check if the error is a 503 capacity error
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            print("Gemini 3.6 Flash busy. Attempting fallback model...")
            try:
                # Fallback to an alternate version or tier
                response = client.models.generate_content(
                    model='gemini-3.1-pro-preview',  # Using Pro as a resilient fallback
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3)
                )
                return response.text if response.text else "[Error: Fallback returned empty response]"
            except Exception as fallback_err:
                return f"[Fallback failed: {str(fallback_err)}]"
                
        return f"[Error processing with Gemini: {str(e)}]"



if __name__ == "__main__":
    # Get user input for file path
    user_input = input("Enter a file path or a paragraph of text: ").strip()
    
    if os.path.exists(user_input):
        print(f"File detected: Loading {user_input}...")
        
        # Select the correct LangChain loader based on file extension
        file_extension = os.path.splitext(user_input)[1].lower()
        if file_extension == ".txt":
            loader = TextLoader(user_input)
        elif file_extension == ".pdf":
            loader = PyPDFLoader(user_input)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(user_input)
        else:
            print("Unsupported file format. Please use .txt, .pdf, or .docx")
            exit()
            
        # Extract text content
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        
        print("\nExtracted Text Content Summary:")
        print(f"Size: {len(full_text)} characters\n")
        
        print("Summarizing with Google Gemini API...")
        summary = main_function(full_text)
        
        print("\n=== Summarized Content ===")
        print(summary)
        
    else:
        print("\nInput is not a valid file path. Processing input directly as raw text...")
        print("Summarizing with Google Gemini API...")
        summary = main_function(user_input)
        
        print("\n=== Summarized Content ===")
        print(summary)
