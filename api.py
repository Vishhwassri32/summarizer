from summary_genmini import main_function
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

app = FastAPI()
app = FastAPI(title="Gemini Document Summarizer API")



class SummaryRequest(BaseModel):
    text: str

@app.post("/summarize-text")
async def summarize_text(payload: SummaryRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
        
    summary = main_function(payload.text)
    return {"summary": summary}


# Endpoint 2: Summarize uploaded documents (.txt, .pdf, .docx) using LangChain loaders
@app.post("/summarize-file")
async def summarize_file(file: UploadFile = File(...)):
    # Create a temporary directory to save the file for LangChain loaders
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file locally
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Select the correct LangChain loader based on file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        elif file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .txt, .pdf, or .docx")

        # Load and extract content
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file contains no readable text.")

        # Get the complete summary
        summary = main_function(full_text)
        return {"filename": file.filename, "summary": summary}

    finally:
        # Clean up temporary file asset after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            