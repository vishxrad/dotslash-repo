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

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_uploaded_file(file):
    global vault_content, vault_embeddings_tensor
    
    try:
        # Read content from uploaded file
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        # Save to vault.txt
        with open("vault.txt", "w", encoding='utf-8') as f:
            f.writelines(content)
        
        # Update vault content
        vault_content = content
        
        # Generate new embeddings
        vault_embeddings = []
        for line in content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=line)
            vault_embeddings.append(response["embedding"])
        
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        
        return f"✓ File uploaded and processed successfully\n- Lines processed: {len(content)}\n- Embeddings generated: {len(vault_embeddings)}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def append_to_vault(file):
    global vault_content, vault_embeddings_tensor
    
    try:
        # Read content from uploaded file
        with open(file.name, 'r', encoding='utf-8') as f:
            new_content = f.readlines()
        
        # Append to vault.txt
        with open("vault.txt", "a", encoding='utf-8') as f:
            f.writelines(new_content)
        
        # Update vault content
        vault_content.extend(new_content)
        
        # Generate new embeddings for appended content
        new_embeddings = []
        for line in new_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=line)
            new_embeddings.append(response["embedding"])
        
        # Concatenate with existing embeddings
        if vault_embeddings_tensor.nelement() == 0:
            vault_embeddings_tensor = torch.tensor(new_embeddings)
        else:
            vault_embeddings_tensor = torch.cat([vault_embeddings_tensor, torch.tensor(new_embeddings)], dim=0)
        
        return f"✓ File appended successfully\n- New lines added: {len(new_content)}\n- Total lines: {len(vault_content)}"
    except Exception as e:
        return f"Error appending file: {str(e)}"

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=20):
    if vault_embeddings.nelement() == 0:
        return []
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
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
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    response_data = {
        "original_query": user_input,
        "rewritten_query": "",
        "context": "",
        "response": "",
        "conversation_history": conversation_history
    }
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        response_data["rewritten_query"] = rewritten_query
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        response_data["context"] = context_str
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    assistant_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_response})
    response_data["response"] = assistant_response
    
    return response_data

def reset_context():
    global conversation_history, vault_embeddings_tensor, vault_content
    conversation_history = []
    vault_content = []
    vault_file_name = "vault.txt"
    if os.path.exists(vault_file_name):
        with open(vault_file_name, "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    return "Context reset successfully"

def initialize_rag(model_name="model"):
    global client, vault_content, vault_embeddings_tensor
    
    status_message = ""
    
    # Initialize Ollama client
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='llama3'
        )
        status_message += "✓ Client initialized successfully\n"
    except Exception as e:
        return f"Failed to initialize client: {str(e)}"
    
    # Load vault content
    try:
        if os.path.exists("vault.txt"):
            with open("vault.txt", "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
        status_message += "✓ Vault content loaded successfully\n"
    except Exception as e:
        return f"Failed to load vault content: {str(e)}"
    
    # Generate embeddings
    try:
        vault_embeddings = []
        for content in vault_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        status_message += "✓ Embeddings generated successfully\n"
    except Exception as e:
        return f"Failed to generate embeddings: {str(e)}"
    
    status_message += "\nSystem is ready!"
    return status_message

def chat_interface(message, history, context_box, rewritten_query_box):
    system_message = """
    You are a helpful assistant that is an expert at extracting the most useful information from a given text. 
    You specialize in assisting with questions and providing relevant information about medical queries and interpreting patients' health reports. 
    Your responses should always bring in extra relevant medical insights, best practices, and explanations of health metrics from outside the given context, if necessary, while focusing on the specific needs of patients and healthcare professionals.
    """
    
    response_data = ollama_chat(
        message,
        system_message,
        vault_embeddings_tensor,
        vault_content,
        "model",
        conversation_history
    )
    
    # Update the context and rewritten query displays
    context_display = "No relevant context found."
    if response_data["context"]:
        context_display = response_data["context"]
    
    rewritten_query_display = "Original query used."
    if response_data["rewritten_query"]:
        rewritten_query_display = response_data["rewritten_query"]
    
    return response_data["response"], context_display, rewritten_query_display

# Create the Gradio interface
def create_gradio_interface():
    # Initialize the RAG system first
    init_status = initialize_rag()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Medical RAG Assistant
        This system helps answer medical queries using a knowledge base of medical information.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your medical query here...",
                    container=False
                )
                # Add location input
                location_input = gr.Textbox(
                    label="Your Location (city, country)",
                    placeholder="e.g., Delhi, India",
                    value=""
                )
                # Add both results boxes
                doctor_results = gr.Markdown(
                    value="No search performed yet.",
                    label="Doctor Search Results"
                )
                medicine_results = gr.Markdown(
                    value="No medicine search performed yet.",
                    label="Medicine Purchase Options"
                )
                # Add buttons below chat
                with gr.Row():
                    doctor_type = gr.Button("Find Specialist")
                    medicine_type = gr.Button("Find Medicine")

            with gr.Column(scale=2):
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
                system_status = gr.Textbox(
                    label="System Status",
                    value=init_status,
                    interactive=False
                )
                clear = gr.Button("Clear Conversation")
        
        def process_upload(file, mode):
            if file is None:
                return "No file selected."
            
            if mode == "Replace existing content":
                return save_uploaded_file(file)
            else:
                return append_to_vault(file)
        
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot(history, context_box, rewritten_query_box):
            user_message = history[-1][0]
            bot_message, context, rewritten = chat_interface(user_message, history, context_box, rewritten_query_box)
            history[-1][1] = bot_message
            return history, context, rewritten
        
        def clear_conversation():
            reset_status = reset_context()
            return (None, 
                    "No context retrieved yet.", 
                    "No query processed yet.",
                    f"Conversation cleared.\n{reset_status}")
        
        # Add new button handlers
        def analyze_doctor_type(vault_content, vault_embeddings_tensor):
            prompt = """Based on the patient's medical history and conditions, identify a single, specific medical specialist that would be most appropriate. Focus on:
            1. The primary medical condition mentioned
            2. The most urgent medical need
            3. The most relevant specialist
            
            Return ONLY the specialist type as a single word (e.g., 'Cardiologist', 'Neurologist'). If there's not enough context to determine a specific specialist, return an empty string."""
            
            # Get relevant context about patient's conditions
            relevant_context = get_relevant_context(prompt, vault_embeddings_tensor, vault_content)
            context_str = "\n".join(relevant_context) if relevant_context else "No medical context found."
            
            messages = [
            {"role": "system", "content": "You are a medical expert. Your task is to identify a single appropriate specialist based on the primary condition in the context. Return ONLY the specialist type as one word. If you cannot determine a specific specialist with confidence, return an empty string."},
            {"role": "user", "content": f"Based on this medical context, name ONE specific specialist type. Return ONLY the specialist type as one word or an empty string if uncertain. Context:\n{context_str}"}
            ]
            
            response = client.chat.completions.create(
                model="model",
                messages=messages,
                max_tokens=500,
            )
            
            return response.choices[0].message.content

        def handle_doctor_type(location):
            if not location:
                return "Please enter your location first.", "Please enter your location first."
            
            specialist_recommendation = analyze_doctor_type(vault_content, vault_embeddings_tensor)
            if not specialist_recommendation:
                return "Could not determine specialist type.", "Could not determine appropriate specialist type from medical context."
            
            search_result = search_doctors(specialist_recommendation, location)
            
            conversation_history.append({
                "role": "user",
                "content": f"Finding {specialist_recommendation}s near {location}"
            })
            conversation_history.append({
                "role": "assistant",
                "content": search_result
            })
            
            return specialist_recommendation, search_result

        def search_doctors(specialist_type, location):
            try:
                if not specialist_type or not location:
                    return "Please provide both specialist type and location."
                
                # Construct search query
                search_query = f"{specialist_type}s near {location}"
                
                # Call Google Search API
                params = {
                    "api_key": "7b60bbdc4ac1aaa076ca3b5b62855901189fe6337688a3e73af8beb958fc323b",
                    "engine": "google",
                    "q": search_query,
                    "location": location,
                    "num": 3  # Get only top result
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Extract top result and format with markdown link
                if "organic_results" in results and results["organic_results"]:
                    top_result = results["organic_results"][0]
                    return f"""Top Result for {search_query}:
Title: {top_result['title']}
Address: {top_result.get('snippet', 'No address available')}
[Click here to view details]({top_result['link']})"""
                else:
                    return f"No results found for {search_query}"
                
            except Exception as e:
                return f"Error performing search: {str(e)}"
            
        def analyze_medicine_type(vault_content, vault_embeddings_tensor):
            prompt = """Based on the patient's medical history and conditions, identify a single, specific medication that would be most appropriate. Focus on:
            1. The primary medical condition mentioned
            2. Standard first-line treatments
            3. Common medication choices
            
            Return ONLY the name of a single medication. If there's not enough context to make a specific recommendation, return an empty string."""
            
            relevant_context = get_relevant_context(prompt, vault_embeddings_tensor, vault_content)
            context_str = "\n".join(relevant_context) if relevant_context else "No medical context found."
            
            messages = [
            {"role": "system", "content": "You are a medical expert. Your task is to identify a single appropriate medication based on the primary condition in the context. Return ONLY the medication name. If you cannot determine a specific medication with confidence, return an empty string."},
            {"role": "user", "content": f"Based on this medical context, name ONE specific medication that would be appropriate. Return ONLY the medication name or an empty string if uncertain. Context:\n{context_str}"}
            ]
            
            response = client.chat.completions.create(
                model="model",
                messages=messages,
                max_tokens=500,
            )
            
            return response.choices[0].message.content

        def search_medicine(medicine_name, location):
            try:
                if not medicine_name:
                    return "Could not determine appropriate medication.", "No medication to search for."
                
                # Construct search query
                search_query = f"buy {medicine_name} online pharmacy"
                
                # Call Google Search API
                params = {
                    "api_key": "7b60bbdc4ac1aaa076ca3b5b62855901189fe6337688a3e73af8beb958fc323b",
                    "engine": "google",
                    "q": search_query,
                    "location": location,
                    "num": 3
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Format results with markdown links
                if "organic_results" in results and results["organic_results"]:
                    result_text = f"Purchase options for {medicine_name}:\n\n"
                    for result in results["organic_results"][:3]:
                        result_text += f"- [{result['title']}]({result['link']})\n"
                        result_text += f"  {result.get('snippet', 'No description available')}\n\n"
                    return medicine_name, result_text
                else:
                    return medicine_name, f"No online purchase options found for {medicine_name}"
                
            except Exception as e:
                return medicine_name, f"Error performing search: {str(e)}"

        def handle_medicine_type(location):
            if not location:
                return "Please enter your location first.", "Please enter your location first."
            
            medication_recommendation = analyze_medicine_type(vault_content, vault_embeddings_tensor)
            if not medication_recommendation:
                return "Could not determine appropriate medication.", "Could not determine appropriate medication from medical context."
            
            medicine_name, search_result = search_medicine(medication_recommendation, location)
            
            conversation_history.append({
                "role": "user",
                "content": f"Finding purchase options for {medicine_name}"
            })
            conversation_history.append({
                "role": "assistant",
                "content": f"Recommended medication: {medicine_name}"
            })
            
            return medicine_name, search_result

        # Add click handlers for both buttons
        doctor_type.click(
            handle_doctor_type,
            inputs=[location_input],
            outputs=[msg, doctor_results]  # Make sure we're using doctor_results
        )

        medicine_type.click(
            handle_medicine_type,
            inputs=[location_input],
            outputs=[msg, medicine_results]
        )

        # Event handlers
        file_upload.upload(
            process_upload,
            inputs=[file_upload, upload_mode],
            outputs=[upload_status]
        )
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, context_box, rewritten_query_box], 
            [chatbot, context_box, rewritten_query_box]
        )
        
        clear.click(clear_conversation, None, 
                   [chatbot, context_box, rewritten_query_box, system_status], 
                   queue=False)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, server_port=7860)
