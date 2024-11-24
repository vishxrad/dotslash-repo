import torch  
import ollama 
import os  
from openai import OpenAI  
import json 
import gradio as gr  
from pathlib import Path  
from serpapi import GoogleSearch  

# Global variables for conversation history and vault content
conversation_history = []  
vault_embeddings_tensor = torch.tensor([])  
vault_content = []  
client = None  

# Function to read the contents of a file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to save uploaded file content to the vault
def save_uploaded_file(file):
    global vault_content, vault_embeddings_tensor
    
    try:
        # Read content from the uploaded file
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        # Save the content to a local file named 'vault.txt'
        with open("vault.txt", "w", encoding='utf-8') as f:
            f.writelines(content)
        
        # Update the global vault content with the new content
        vault_content = content
        
        # Generate embeddings for each line in the uploaded content
        vault_embeddings = []
        for line in content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=line)
            vault_embeddings.append(response["embedding"])
        
        # Convert the list of embeddings into a tensor
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        
        return f"✓ File uploaded and processed successfully\n- Lines processed: {len(content)}\n- Embeddings generated: {len(vault_embeddings)}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Function to append new content to the existing vault
def append_to_vault(file):
    global vault_content, vault_embeddings_tensor
    
    try:
        # Read content from the uploaded file
        with open(file.name, 'r', encoding='utf-8') as f:
            new_content = f.readlines()
        
        # Append the new content to the 'vault.txt' file
        with open("vault.txt", "a", encoding='utf-8') as f:
            f.writelines(new_content)
        
        # Update the global vault content
        vault_content.extend(new_content)
        
        # Generate embeddings for the new content
        new_embeddings = []
        for line in new_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=line)
            new_embeddings.append(response["embedding"])
        
        # Merge new embeddings with the existing tensor
        if vault_embeddings_tensor.nelement() == 0:  # Check if the tensor is empty
            vault_embeddings_tensor = torch.tensor(new_embeddings)
        else:
            vault_embeddings_tensor = torch.cat([vault_embeddings_tensor, torch.tensor(new_embeddings)], dim=0)
        
        return f"✓ File appended successfully\n- New lines added: {len(new_content)}\n- Total lines: {len(vault_content)}"
    except Exception as e:
        return f"Error appending file: {str(e)}"

# Function to retrieve relevant context from the vault based on input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=20):
    if vault_embeddings.nelement() == 0:  # Check if the vault embeddings tensor is empty
        return []
    
    # Generate embedding for the rewritten input query
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    
    # Calculate cosine similarity between input embedding and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    
    # Get top-k indices of most similar embeddings
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    
    # Retrieve the corresponding content for the top-k indices
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to rewrite a user query based on the conversation history
def rewrite_query(user_input_json, conversation_history, ollama_model):
    # Extract the query text from the JSON input
    user_input = json.loads(user_input_json)["Query"]
    
    # Compile the recent conversation history into a single string
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    
    # Prepare a prompt for the model to rewrite the query
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    
    # Generate the rewritten query using the specified model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    
    # Extract and return the rewritten query as a JSON object
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})





def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    """
    Handles the AI conversation using the Ollama model.
    """
    # Add user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Initialize response data
    response_data = {
        "original_query": user_input,
        "rewritten_query": "",
        "context": "",
        "response": "",
        "conversation_history": conversation_history
    }

    if len(conversation_history) > 1:
        # Rewrite query for better understanding
        query_json = {"Query": user_input, "Rewritten Query": ""}
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        response_data["rewritten_query"] = rewritten_query
    else:
        rewritten_query = user_input

    # Retrieve relevant context from the vault
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        # Combine relevant context into a string
        context_str = "\n".join(relevant_context)
        response_data["context"] = context_str

    # Append context to user input if available
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str

    # Update conversation history with context
    conversation_history[-1]["content"] = user_input_with_context

    # Construct messages for the model
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    # Get response from the Ollama model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )

    # Extract assistant's response
    assistant_response = response.choices[0].message.content

    # Append response to conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})
    response_data["response"] = assistant_response

    return response_data


def reset_context():
    """
    Resets the conversation context and reloads vault data.
    """
    global conversation_history, vault_embeddings_tensor, vault_content

    # Clear conversation history and vault content
    conversation_history = []
    vault_content = []

    # Load vault content if the file exists
    vault_file_name = "vault.txt"
    if os.path.exists(vault_file_name):
        with open(vault_file_name, "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

    # Generate embeddings for vault content
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])

    # Convert embeddings to a tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)

    return "Context reset successfully"


def initialize_rag(model_name="model"):
    """
    Initializes the RAG system, including the Ollama client, vault content, and embeddings.
    """
    global client, vault_content, vault_embeddings_tensor

    status_message = ""

    # Initialize the Ollama client
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='llama3'
        )
        status_message += "✓ Client initialized successfully\n"
    except Exception as e:
        return f"Failed to initialize client: {str(e)}"

    # Load vault content from file
    try:
        if os.path.exists("vault.txt"):
            with open("vault.txt", "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
        status_message += "✓ Vault content loaded successfully\n"
    except Exception as e:
        return f"Failed to load vault content: {str(e)}"

    # Generate embeddings for the vault content
    try:
        vault_embeddings = []
        for content in vault_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        status_message += "✓ Embeddings generated successfully\n"
    except Exception as e:
        return f"Failed to generate embeddings: {str(e)}"

    # Return final status
    status_message += "\nSystem is ready!"
    return status_message


def chat_interface(message, history, context_box, rewritten_query_box):
    """
    Handles user interaction by processing input through the assistant and updating the UI.
    """
    # Define the system's role and behavior
    system_message = """
    You are a helpful assistant specializing in extracting useful information from text.
    Focus on medical queries, health reports, and providing insights for patients and professionals.
    Include relevant medical practices and metrics if needed.
    """

    # Get the assistant's response using the chat function
    response_data = ollama_chat(
        message,
        system_message,
        vault_embeddings_tensor,
        vault_content,
        "model",
        conversation_history
    )

    # Update the context box with relevant context or a default message
    context_display = "No relevant context found."
    if response_data["context"]:
        context_display = response_data["context"]

    # Update the rewritten query box with the rewritten query or default text
    rewritten_query_display = "Original query used."
    if response_data["rewritten_query"]:
        rewritten_query_display = response_data["rewritten_query"]

    return response_data["response"], context_display, rewritten_query_display

# Create the Gradio interface
def create_gradio_interface():
    # Initialize the RAG system first
    init_status = initialize_rag() 
    # Gradio Blocks interface setup
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Medical RAG Assistant
        This system helps answer medical queries using a knowledge base of medical information.
        """)  

        with gr.Row():
            with gr.Column(scale=3):
                # Chatbot interface with input field for user query
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your medical query here...",
                    container=False
                )
                # Input box for user to provide location
                location_input = gr.Textbox(
                    label="Your Location (city, country)",
                    placeholder="e.g., Delhi, India",
                    value=""
                )
                # Markdown for displaying doctor search results
                doctor_results = gr.Markdown(
                    value="No search performed yet.",
                    label="Doctor Search Results"
                )
                # Markdown for displaying medicine search results
                medicine_results = gr.Markdown(
                    value="No medicine search performed yet.",
                    label="Medicine Purchase Options"
                )
                # Buttons for additional functionality
                with gr.Row():
                    doctor_type = gr.Button("Find Specialist") 
                    medicine_type = gr.Button("Find Medicine")  

            with gr.Column(scale=2):
                # Knowledge base management section
                with gr.Accordion("Knowledge Base Management", open=True):
                    file_upload = gr.File(
                        label="Upload Text File",
                        file_types=[".txt"],  
                        file_count="single"
                    )
                    upload_mode = gr.Radio(
                        choices=["Replace existing content", "Append to existing content"],
                        value="Replace existing content",
                        label="Upload Mode"
                    )
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        value="No file uploaded yet.",
                        interactive=False
                    )

                # Query processing details section
                with gr.Accordion("Query Processing Details", open=True):
                    rewritten_query_box = gr.Textbox(
                        label="Rewritten Query",
                        value="No query processed yet.",
                        interactive=False
                    )
                    context_box = gr.Textbox(
                        label="Retrieved Context",
                        value="No context retrieved yet.",
                        interactive=False,
                        lines=10  
                    )
                # Status of the system (initialized or not)
                system_status = gr.Textbox(
                    label="System Status",
                    value=init_status,
                    interactive=False
                )
                clear = gr.Button("Clear Conversation") 

        # File upload handler
        def process_upload(file, mode):
            if file is None:
                return "No file selected."
            
            if mode == "Replace existing content":
                return save_uploaded_file(file) 
            else:
                return append_to_vault(file) 

        # User message input handling
        def user(user_message, history):
            return "", history + [[user_message, None]]  

        # Bot response generation
        def bot(history, context_box, rewritten_query_box):
            user_message = history[-1][0]
            bot_message, context, rewritten = chat_interface(user_message, history, context_box, rewritten_query_box)
            history[-1][1] = bot_message 
            return history, context, rewritten

        # Clear conversation history and reset context
        def clear_conversation():
            reset_status = reset_context()
            return (None, 
                    "No context retrieved yet.", 
                    "No query processed yet.",
                    f"Conversation cleared.\n{reset_status}")

        # Specialist analysis based on medical context
        def analyze_doctor_type(vault_content, vault_embeddings_tensor):
            # Prompt for analyzing and identifying the appropriate medical specialist
            prompt = """Based on the patient's medical history and conditions, identify a single, specific medical specialist that would be most appropriate. Focus on:
            1. The primary medical condition mentioned
            2. The most urgent medical need
            3. The most relevant specialist
            
            Return ONLY the specialist type as a single word (e.g., 'Cardiologist', 'Neurologist'). If there's not enough context to determine a specific specialist, return an empty string."""
            
            # Retrieve relevant medical context
            relevant_context = get_relevant_context(prompt, vault_embeddings_tensor, vault_content)
            context_str = "\n".join(relevant_context) if relevant_context else "No medical context found."
            
            # Prepare system and user messages for the model
            messages = [
            {"role": "system", "content": "You are a medical expert. Your task is to identify a single appropriate specialist based on the primary condition in the context. Return ONLY the specialist type as one word. If you cannot determine a specific specialist with confidence, return an empty string."},
            {"role": "user", "content": f"Based on this medical context, name ONE specific specialist type. Return ONLY the specialist type as one word or an empty string if uncertain. Context:\n{context_str}"}
            ]
            
            # Generate response using a language model
            response = client.chat.completions.create(
                model="model",
                messages=messages,
                max_tokens=500,
            )
            
            return response.choices[0].message.content

        # Doctor search based on specialist type and location
        def handle_doctor_type(location):
            if not location:
                return "Please enter your location first.", "Please enter your location first."
            
            # Get specialist recommendation
            specialist_recommendation = analyze_doctor_type(vault_content, vault_embeddings_tensor)
            if not specialist_recommendation:
                return "Could not determine specialist type.", "Could not determine appropriate specialist type from medical context."
            
            # Perform Google search for doctors
            search_result = search_doctors(specialist_recommendation, location)
            
            def search_doctors(specialist, location):
                search_params = {
                    "q": f"{specialist} near {location}",
                    "location": location,
                    "hl": "en",
                    "gl": "us",
                    "api_key": "YOUR_SERPAPI_KEY"
                }
                search = GoogleSearch(search_params)
                results = search.get_dict()
                return json.dumps(results, indent=2)
            
            conversation_history.append({
                "role": "user",
                "content": f"Finding {specialist_recommendation}s near {location}"
            })
            conversation_history.append({
                "role": "assistant",
                "content": search_result
            })
            
            return specialist_recommendation, search_result

       

        # Function to search for medicines based on location
        def search_medicines(location):
            search_params = {
                "q": "pharmacy near " + location,
                "location": location,
                "hl": "en",
                "gl": "us",
                "api_key": "YOUR_SERPAPI_KEY"
            }
            search = GoogleSearch(search_params)
            results = search.get_dict()
            return json.dumps(results, indent=2)

        # Medicine search based on location
        def handle_medicine_type(location):
            if not location:
                return "Please enter your location first.", "Please enter your location first."
            
            # Perform Google search for medicines
            search_result = search_medicines(location)
            
            conversation_history.append({
                "role": "user",
                "content": f"Finding medicines near {location}"
            })
            conversation_history.append({
                "role": "assistant",
                "content": search_result
            })
            
            return "Medicine search", search_result
        
        # Attach event handlers to UI components
        doctor_type.click(
            handle_doctor_type,
            inputs=[location_input],
            outputs=[msg, doctor_results]  
        )

        medicine_type.click(
            handle_medicine_type,
            inputs=[location_input],
            outputs=[msg, medicine_results]
        )

        # File upload event
        file_upload.upload(
            process_upload,
            inputs=[file_upload, upload_mode],
            outputs=[upload_status]
        )
        
        # Input text submission handling
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, context_box, rewritten_query_box], 
            [chatbot, context_box, rewritten_query_box]
        )
        
        clear.click(clear_conversation, None, 
                   [chatbot, context_box, rewritten_query_box, system_status], 
                   queue=False)
    
    return demo

# Launch the Gradio app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, server_port=7860)
