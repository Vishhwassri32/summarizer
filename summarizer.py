import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Initialize the Model Locally optimized for Mac
model_id = "microsoft/Phi-3-mini-4k-instruct"

print("Loading model and tokenizer locally...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the proper precision format for Mac (float16 for MPS GPU, float32 for CPU fallback)
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
    print("-> Using Apple Silicon GPU Acceleration (MPS)")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("-> Apple Silicon GPU not found. Falling back to CPU.")

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
) # Explicitly map the model to your Mac hardware

# 2. Wrap the Hugging Face pipeline into LangChain
hf_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=300, 
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# 2. File and Text Loader Handler
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

# =====================================================================
# 3. Test Inputs (Uncomment whichever one you want to use)
# =====================================================================

# Option A: Testing with a raw text paragraph
my_input = """
Artificial Intelligence is advancing at a breakneck pace. Large language models are transforming 
software development, automated content creation, and data processing. Developers are now 
leveraging these engines to build smart workflows that extract key data points seamlessly. However, 
running these architectures requires either expensive cloud setups or optimized local hardware environments. 
"""

# Option B: Testing with a file path (Uncomment to use)
# my_input = "huge_report.pdf" 

# =====================================================================

try:
    # 4. Fetch the final text string
    raw_text = get_text_input(my_input)
    
    # 5. Chunking (Safe for both short paragraphs and 81-page files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    chunks = text_splitter.split_text(raw_text)
    print(f"Total chunks to process: {len(chunks)}")
    
    # 6. Map Step: Summarise chunks
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        map_prompt = f"Summarise the key points of this section concisely:\n\n{chunk}"
        chunk_summary = llm.invoke(map_prompt)
        partial_summaries.append(chunk_summary)
        print(f"-> Processed chunk {i+1}/{len(chunks)}")
        
    # 7. Reduce Step: Final processing
    print("\nGenerating final master summary...")
    combined_context = "\n\n".join(partial_summaries)
        
    reduce_prompt = (
        "You are an expert executive assistant. Provide a highly professional, cohesive executive summary "
        "based on the following section summaries. Use bullet points for key takeaways:\n\n"
        f"{combined_context}"
    )
    
    final_summary = llm.invoke(reduce_prompt)
    
    print("\n=================== FINAL SUMMARY ===================")
    print(final_summary)

except Exception as e:
    print(f"\nAn error occurred: {e}")
