import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# Global variables for conversation history and vault content
conversation_history = []
vault_embeddings_tensor = torch.tensor([])  # Initially empty
vault_content = []

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=5):
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
    
    return json.dumps(response_data, ensure_ascii=False, indent=2)

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
    return json.dumps({"status": "success", "message": "Context reset successfully"})

def initialize_rag(model_name="test2"):
    global client, vault_content, vault_embeddings_tensor
    
    initialization_status = {
        "status": "initializing",
        "steps": {}
    }
    
    # Initialize Ollama client
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='llama3'
        )
        initialization_status["steps"]["client_initialization"] = "success"
    except Exception as e:
        initialization_status["steps"]["client_initialization"] = f"failed: {str(e)}"
        return json.dumps(initialization_status)
    
    # Load vault content
    try:
        if os.path.exists("vault.txt"):
            with open("vault.txt", "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
        initialization_status["steps"]["vault_loading"] = "success"
    except Exception as e:
        initialization_status["steps"]["vault_loading"] = f"failed: {str(e)}"
        return json.dumps(initialization_status)
    
    # Generate embeddings
    try:
        vault_embeddings = []
        for content in vault_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        initialization_status["steps"]["embeddings_generation"] = "success"
    except Exception as e:
        initialization_status["steps"]["embeddings_generation"] = f"failed: {str(e)}"
        return json.dumps(initialization_status)
    
    initialization_status["status"] = "ready"
    return json.dumps(initialization_status)

if __name__ == "__main__":
    system_message = """
    You are a helpful assistant that is an expert at extracting the most useful information from a given text. 
    You specialize in assisting with questions and providing relevant information about medical queries and interpreting patients' health reports. 
    Your responses should always bring in extra relevant medical insights, best practices, and explanations of health metrics from outside the given context, if necessary, while focusing on the specific needs of patients and healthcare professionals.
    """
    
    # Initialize the RAG system
    init_status = initialize_rag()
    print(init_status)
    
    # Example of how to use the modified functions
    while True:
        user_input = input("Enter query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        response_json = ollama_chat(
            user_input,
            system_message,
            vault_embeddings_tensor,
            vault_content,
            "test2",
            conversation_history
        )
        print(response_json)