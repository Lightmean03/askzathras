from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def diagnose_document_parsing():
    """Comprehensive document parsing diagnostics"""
    print("="*60)
    print("DOCUMENT PARSING DIAGNOSTICS")
    print("="*60)
    
    try:
        # Load the document
        print("📄 Loading PDF document...")
        loader = PyPDFLoader("./content/book.pdf")
        documents = loader.load()
        
        print(f"✅ Successfully loaded PDF")
        print(f"   📊 Total pages: {len(documents)}")
        
        # Check if document loaded properly
        if not documents:
            print("❌ ERROR: No documents loaded! Check file path.")
            return False
        
        # Examine first page
        print(f"\n📖 FIRST PAGE ANALYSIS:")
        first_doc = documents[0]
        print(f"   📝 Content length: {len(first_doc.page_content)} characters")
        print(f"   🏷️  Metadata: {first_doc.metadata}")
        
        # Show sample content
        print(f"\n📄 FIRST 300 CHARACTERS:")
        sample_content = first_doc.page_content[:300]
        print(f"'{sample_content}{'...' if len(first_doc.page_content) > 300 else ''}'")
        
        # Check for common parsing issues
        if len(first_doc.page_content.strip()) == 0:
            print("❌ WARNING: First page appears to be empty!")
        
        if len(first_doc.page_content) < 50:
            print("⚠️  WARNING: First page content seems very short - possible parsing issue")
        
        # Check all pages for content
        empty_pages = 0
        short_pages = 0
        page_lengths = []
        
        for i, doc in enumerate(documents):
            content_length = len(doc.page_content.strip())
            page_lengths.append(content_length)
            
            if content_length == 0:
                empty_pages += 1
                print(f"❌ Page {i+1} is empty!")
            elif content_length < 100:
                short_pages += 1
                print(f"⚠️  Page {i+1} is very short ({content_length} chars)")
        
        print(f"\n📊 PAGE CONTENT STATISTICS:")
        print(f"   Empty pages: {empty_pages}")
        print(f"   Short pages (<100 chars): {short_pages}")
        print(f"   Average page length: {sum(page_lengths)/len(page_lengths):.1f} characters")
        print(f"   Min page length: {min(page_lengths)} characters")
        print(f"   Max page length: {max(page_lengths)} characters")
        
        return documents
        
    except FileNotFoundError:
        print("❌ ERROR: PDF file not found! Check the file path:")
        print("   Current path: './content/Chapter-1-Students.pdf'")
        return False
    except Exception as e:
        print(f"❌ ERROR loading PDF: {e}")
        return False

def diagnose_text_splitting(documents):
    """Diagnose text splitting process"""
    print("\n" + "="*60)
    print("TEXT SPLITTING DIAGNOSTICS")
    print("="*60)
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50, 
        separators=["\n\n", "\n", " ", ""]  # Better separators for content structure
    )
    docs = text_splitter.split_documents(documents=documents)
    
    print(f"✅ Successfully split documents")
    print(f"   📊 Total chunks: {len(docs)}")
    print(f"   📊 Chunks per page: {len(docs)/len(documents):.1f}")
    
    # Analyze chunk sizes
    chunk_sizes = [len(doc.page_content) for doc in docs]
    print(f"\n📊 CHUNK SIZE ANALYSIS:")
    print(f"   Average size: {sum(chunk_sizes)/len(chunk_sizes):.1f} characters")
    print(f"   Min size: {min(chunk_sizes)} characters")
    print(f"   Max size: {max(chunk_sizes)} characters")
    print(f"   Target size: 400 characters")
    
    # Check for empty chunks
    empty_chunks = [i for i, size in enumerate(chunk_sizes) if size == 0]
    if empty_chunks:
        print(f"❌ WARNING: {len(empty_chunks)} empty chunks found at indices: {empty_chunks}")
    
    # Show first few chunks
    print(f"\n📄 FIRST 3 CHUNKS PREVIEW:")
    for i, chunk in enumerate(docs[:3]):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Page: {chunk.metadata.get('page', 'unknown')}")
        print(f"Content preview: '{chunk.page_content[:100]}{'...' if len(chunk.page_content) > 100 else ''}'")
    
    return docs

def test_retrieval(retriever, docs):
    """Test the retrieval system"""
    print("\n" + "="*60)
    print("RETRIEVAL SYSTEM TEST")
    print("="*60)
    
    # Test queries that should definitely be in the document
    test_queries = [
        "Cyber operations",
        "Towson University COSC 431",
        "Initial Access",
        "testing laboratory",
        "virtual machines",
        "Metasploit",
        "Windows and Linux",
        "Chapter 1"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        try:
            relevant_docs = retriever.invoke(query)  # Using new invoke method
            print(f"   Found {len(relevant_docs)} relevant chunks")
            
            if relevant_docs:
                # Show the most relevant chunk
                top_chunk = relevant_docs[0]
                print(f"   Page: {top_chunk.metadata.get('page', 'unknown')}")
                print(f"   Top result preview: '{top_chunk.page_content[:150]}{'...' if len(top_chunk.page_content) > 150 else ''}'")
            else:
                print("   ❌ No relevant documents found!")
                
        except Exception as e:
            print(f"   ❌ Error during retrieval: {e}")

# Main execution with diagnostics
print("Starting RAG system with full diagnostics...")

# Step 1: Diagnose document loading
documents = diagnose_document_parsing()
if not documents:
    print("Cannot continue - document loading failed!")
    exit(1)

# Step 2: Diagnose text splitting
docs = diagnose_text_splitting(documents)

print("\n" + "="*60)
print("EMBEDDING AND VECTOR STORE CREATION")
print("="*60)

print("Creating embeddings...")

# Load embedding with cpu - using different model for better retrieval
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )
    print("✅ Embeddings model loaded successfully")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    exit(1)

print("Building vector store...")

try:
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("✅ Vector store created successfully")
    
    # Test the vector store immediately with known content
    print("\n🔍 Testing vector store with known content...")
    test_queries = [
        "Cyber operations",
        "Towson University",
        "Initial Access", 
        "testing laboratory",
        "Michael Oleary"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Testing: '{query}'")
        test_docs = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(test_docs):
            print(f"  Result {i+1}: Page {doc.metadata.get('page', 'unknown')}")
            print(f"  Content preview: {doc.page_content[:80]}...")
    
    # Save and reload the vector store
    vectorstore.save_local("faiss_index_")
    persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)
    print("\n✅ Vector store saved and reloaded successfully")
    
    # Create a retriever with better parameters
    retriever = persisted_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Get top 4 most relevant chunks
    )
    print("✅ Retriever created successfully")
    
except Exception as e:
    print(f"❌ Error with vector store: {e}")
    exit(1)

# Step 3: Test retrieval system
test_retrieval(retriever, docs)

print("\n" + "="*60)
print("LLAMA MODEL INITIALIZATION")
print("="*60)

print("Initializing LLaMA model...")

# Initialize the LLaMA model
try:
    llm = OllamaLLM(model="llama3.2:3b")
    print("✅ LLaMA model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading LLaMA model: {e}")
    print("Make sure Ollama is running: ollama serve")
    print("And the model is pulled: ollama pull llama3.2:3b")
    exit(1)

# Create RetrievalQA
try:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("✅ RetrievalQA chain created successfully")
except Exception as e:
    print(f"❌ Error creating QA chain: {e}")
    exit(1)

print("\n" + "="*60)
print("RAG SYSTEM READY!")
print("="*60)
print(f"✅ Document parsed: {len(documents)} pages, {len(docs)} chunks")
print(f"✅ Vector store: {vectorstore.index.ntotal} vectors")
print(f"✅ Model: llama3.2:3b ready")
print("="*60)

# Interactive query loop with enhanced error handling
print("\nAvailable commands:")
print("• Just type your question normally")
print("• 'debug [question]' - See retrieval details")
print("• 'page [number]' - View specific page content")
print("• 'test' - Run quick retrieval tests")
print("• 'exit' - Quit the program")
print("-" * 60)

while True:
    query = input("\n🤖 Enter command or question: ")
    
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    elif query.lower() == "test":
        # Run quick tests
        print("\n🧪 Running quick retrieval tests...")
        test_retrieval(retriever, docs)
    
    elif query.lower().startswith("page "):
        # Page-specific search
        try:
            page_num = int(query.split()[1])
            page_chunks = [doc for doc in docs if doc.metadata.get('page') == page_num - 1]  # PDF pages are 0-indexed
            if page_chunks:
                print(f"\n📄 Found {len(page_chunks)} chunks from page {page_num}:")
                for i, chunk in enumerate(page_chunks):
                    print(f"\n--- PAGE {page_num} CHUNK {i+1} ---")
                    print(chunk.page_content)
                    print("-" * 40)
            else:
                print(f"❌ No content found for page {page_num}")
        except (ValueError, IndexError):
            print("❌ Please use format: page [number] (e.g., 'page 1')")
    
    elif query.lower().startswith("debug "):
        # Debug mode - show retrieval details
        debug_query = query[6:]  # Remove "debug " prefix
        print(f"\n🔍 DEBUG MODE - Query: '{debug_query}'")
        
        try:
            # Show retrieved documents
            relevant_docs = retriever.invoke(debug_query)  # Using new invoke method
            print(f"📊 Retrieved {len(relevant_docs)} documents:")
            
            for i, doc in enumerate(relevant_docs):
                print(f"\n--- RETRIEVED DOC {i+1} ---")
                print(f"Page: {doc.metadata.get('page', 'unknown')}")
                print(f"Content: {doc.page_content}")
                print("-" * 40)
            
            # Generate answer
            result = qa.invoke({"query": debug_query})  # Using new invoke method
            answer = result.get("result", "No answer generated")
            print(f"\n🤖 FINAL ANSWER: {answer}")
            
        except Exception as e:
            print(f"❌ Error in debug mode: {e}")
    
    else:
        # Normal mode
        try:
            print("🔍 Searching and generating response...")
            result = qa.invoke({"query": query})  # Using new invoke method
            answer = result.get("result", "No answer generated")
            print(f"\n🤖 Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("Try using 'debug [your question]' to see more details")
    
    print("-" * 60)
