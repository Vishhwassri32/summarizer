import os
from google import genai
from google.genai import types
import dotenv
# LangChain community loaders and splitters
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. MODEL INITIALIZATION
# ==========================================

dotenv.load_dotenv()  # Load environment variables from .env file

# Initialize Google Gemini Client 
# (Will automatically read the GEMINI_API_KEY environment variable)
client = genai.Client()

# ==========================================
# 2. UTILITY & TEXT PROCESSING FUNCTIONS
# ==========================================

def get_text_input(user_input):
    """
    Checks if the input is a valid file path. 
    If yes, it reads the file. If no, it treats the input as direct text.
    """
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
    else:
        print("Raw text paragraph detected...")
        return user_input


def chunk_text(text, chunk_size=4000, overlap=400):
    """
    Splits the text into smaller chunks for processing.
    Note: Chunk size is increased since Gemini handles large text easily.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", ""]
    )
    return text_splitter.split_text(text)


# ==========================================
# 3. SUMMARIZATION FUNCTION
# ==========================================

def summarize_text_gemini(chunks):
    """
    Summarizes each chunk of text using Google Gemini API.
    """
    summaries = []
    print("\nSummarizing with Google Gemini API...")
    
    for i, chunk in enumerate(chunks):
        prompt = f"Provide a concise summary of the following text chunk:\n\n{chunk}"
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                )
            )
            summaries.append(response.text.strip())
        except Exception as e:
            print(f"Error processing chunk {i+1} with Gemini: {e}")
            summaries.append(f"[Error summarizing chunk {i+1}]")
            
    return "\n\n".join(summaries)


# ==========================================
# 4. EXECUTION FLOW
# ==========================================




# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main_function(user_input):
        # Get user input and split into chunks
        text_content = get_text_input(user_input)
        chunks = chunk_text(text_content)
        
        print(f"\nTotal chunks to process: {len(chunks)}")
        
        # Summarize the chunks using Google Gemini
        summaries = summarize_text_gemini(chunks)
        
        print("\n=================== FINAL SUMMARY ===================")
        print(summaries)




    