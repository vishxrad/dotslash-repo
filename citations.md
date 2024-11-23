# Comprehensive Medical Q&A Dataset  

## **Overview**  
The **Comprehensive Medical Q&A Dataset** is an excellent resource for creating AI systems tailored to the healthcare domain. It is particularly useful for:  
- **Medical-related Machine Learning.**  
- **Natural Language Processing (NLP).**  
- **Question-Answering (Q&A) Systems.**

---

## **Why Use This Dataset?**  
This dataset serves as a **robust foundation** for several applications, such as:  

- **Training AI Models:** Enable AI systems to understand and respond to medical queries.  
- **Simulating Virtual Healthcare Assistants:** Build chatbots to automate healthcare support.  
- **Educational Tools Development:** Create interactive platforms for medical learning.  

---

## **Potential Use Cases**  
1. **AI Chatbots:**  
   Build conversational agents capable of addressing healthcare-related queries.  

2. **Disease Diagnosis Models:**  
   Train AI systems to suggest potential diagnoses based on symptoms.  

3. **Medical Education:**  
   Create Q&A-based study material or simulations for healthcare professionals and students.  

---

## **How We Used the Dataset**  

### **Step 1: Preparing the Dataset**  
We downloaded a medical Q&A dataset from Kaggle, reformatted it using a custom **JavaScript script**, and uploaded it to **Hugging Face** for compatibility with state-of-the-art NLP tools and frameworks.

### **Step 2: Loading the Dataset**  
The dataset is structured into pairs of questions (inputs) and answers (outputs).  
Here’s how to load it using the Hugging Face `datasets` library:  
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("visharxd/medical-qna")

# Access the training split
train_data = dataset["train"]

# View the first example
print(train_data[0])

```

## **Step 3: Loading the Model**
Use the Unsloth library to load the Meta-Llama-3.1-8B model:
```python
from unsloth import FastLanguageModel

# Load the model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    load_in_4bit=True
)
```
Enable 4-bit quantization to reduce memory usage and make training efficient.


## **Step 4: Applying LoRA (Low-Rank Adaptation)**
Fine-tune specific layers of the model using LoRA to reduce computational overhead:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0
)
```
LoRA focuses on specific layers like q_proj, k_proj, v_proj, etc., to improve efficiency.

## **Step 5: Formatting the Dataset**
Format the dataset into Alpaca-style prompts for instruction-based fine-tuning.
Prompt Template:
```python
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
<instruction>

### Input:
<input>

### Response:
<response>
```
Use a custom function to map each dataset example to this format:
```python

alpaca_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

def formatting_prompts_func(examples):
    instructions = examples["context"]
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return {"text": texts}
```
## **Step 6: Fine-Tuning the Model**
Use Hugging Face's SFTTrainer (Supervised Fine-Tuning Trainer) to fine-tune the model:
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
)
```
Configure training parameters:
Batch Size: 2 per device, gradient accumulation for 4 steps (effective batch size = 8).
Learning Rate: 2e-4.

## **Steps: Train for 1000 optimization steps.**

Optimizer: adamw_8bit for memory efficiency.
## **Step 7: Training and Saving the Model**
Train the model:
```python
trainer_stats = trainer.train()
```
Save the fine-tuned model to the outputs directory for future use.
Purpose and Benefits
The fine-tuned model is capable of:

Excelling in Instruction-Based Q&A Tasks:
Generates accurate and contextual responses for medical queries.
Enhancing Healthcare AI Tools:
Suitable for chatbots, education platforms, and automated assistants.
Optimizing Training Efficiency:
Utilizes LoRA and 4-bit quantization to save memory and reduce costs.


# Modern Web Application with React, Tailwind CSS, and Claude AI

A modern, responsive web application built with React, styled with Tailwind CSS, and powered by Claude AI integration.

## 🚀 Features

- Modern React application using latest best practices
- Responsive design with Tailwind CSS utility classes
- AI-powered features using Claude 3 Sonnet
- Fast development workflow with hot module replacement
- Production-ready build configuration
- SEO-friendly structure
- Cross-browser compatibility

## 📋 Prerequisites

Before you begin, ensure you have installed:
- Node.js (v18.0.0 or higher)
- npm (v8.0.0 or higher)
- Git

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the root directory and add your Claude AI API key:
```env
VITE_CLAUDE_API_KEY=your_api_key_here
```

## 🏃‍♂️ Development

Start the development server:
```bash
npm run dev
```

Your application will be available at `http://localhost:5173`

## 🏗️ Build

Create a production build:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## 🎨 Styling

This project uses Tailwind CSS for styling. The configuration file is located at `tailwind.config.js`.

Key styling features:
- Custom color palette
- Responsive breakpoints
- Dark mode support
- Custom component classes

## 🤖 Claude AI Integration

This project integrates with Claude 3 Sonnet for AI features:

1. Set up your Claude AI credentials in the `.env` file
2. Use the provided utility functions in `src/utils/claude.js`
3. Check the API documentation for available endpoints and features

## 📁 Project Structure

```
src/
├── components/     # Reusable React components
├── pages/         # Page components
├── layouts/       # Layout components
├── utils/         # Utility functions
├── services/      # API services
├── hooks/         # Custom React hooks
├── context/       # React context providers
├── styles/        # Global styles and Tailwind CSS
└── assets/        # Static assets
```

## 🧪 Testing

Run tests:
```bash
npm run test
```

Run tests in watch mode:
```bash
npm run test:watch
```

## 📚 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Create production build
- `npm run preview` - Preview production build
- `npm run test` - Run tests
- `npm run lint` - Lint code
- `npm run format` - Format code with Prettier

## 🔧 Configuration

### Tailwind CSS

Tailwind configuration is in `tailwind.config.js`. Customize:
- Theme
- Colors
- Typography
- Breakpoints
- Plugins

### Vite

Vite configuration is in `vite.config.js`. Configure:
- Build options
- Development server
- Plugins
- Aliases

## 📱 Responsive Design

The application is responsive across devices:
- Mobile: 320px and up
- Tablet: 768px and up
- Desktop: 1024px and up
- Large Desktop: 1280px and up

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [React](https://react.dev/) - UI library
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [Claude AI](https://claude.ai/) - AI integration
- [Vite](https://vitejs.dev/) - Build tool

- # Django Backend API

A robust backend API built with Django REST Framework, featuring authentication, database integration, and AI-powered features using ChatGPT.

## 🚀 Features

- RESTful API endpoints
- JWT Authentication
- PostgreSQL database integration
- ChatGPT integration for intelligent responses
- API documentation with Swagger/OpenAPI
- Comprehensive test coverage
- Rate limiting and security features
- Containerized with Docker

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Docker (optional)
- virtualenv or pipenv

## 🛠️ Installation

### Local Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file in the root directory:
```env
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
CHATGPT_API_KEY=your-chatgpt-api-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Create superuser:
```bash
python manage.py createsuperuser
```

### Docker Setup

1. Build the Docker image:
```bash
docker-compose build
```

2. Run the containers:
```bash
docker-compose up
```

## 🏃‍♂️ Development

Start the development server:
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000`

## 🔐 Authentication

The API uses JWT (JSON Web Tokens) for authentication:

1. Obtain token:

# Medical QA AI System

A sophisticated medical question-answering system built using Llama 2 and RAG (Retrieval-Augmented Generation) architecture, fine-tuned on comprehensive medical data.

## 🔍 Overview

This project implements a medical question-answering system using:
- Base Model: Llama 3.1 8b Instruct
- Dataset: Comprehensive Medical Q&A Dataset from Kaggle
- Fine-tuning: Unsloth optimization framework
- RAG Implementation: Local RAG with ChromaDB
- Development Environment: Google Colab
- Model Hosting: Hugging Face Hub

## 🎯 Model Architecture

### Base Model: Llama 3.1 8b Instruct
- Authors: Meta AI Research Team
  - Lead Authors: Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al.
- Size: 8 billion parameters
- Architecture: Transformer-based language model
- Notable Features:
  - Instruction-tuned variant
  - Enhanced medical domain understanding
  - Optimized for dialogue interactions

## 📊 Dataset

### Medical Q&A Dataset
- Source: Kaggle
- URL: [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)
- Statistics:
  - Number of QA pairs: [Your dataset size]
  - Topics covered: [Main medical categories]
  - Languages: English
  - Format: JSON/CSV

### Data Processing
1. Cleaning and Preprocessing
```python
# Example data preprocessing pipeline
def preprocess_data(data):
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Clean text
    data['question'] = data['question'].apply(clean_text)
    data['answer'] = data['answer'].apply(clean_text)
    
    return data
```

2. Data Splitting
- Training: 80%
- Validation: 10%
- Testing: 10%

## 🛠️ Fine-tuning

### Environment Setup
1. Google Colab Configuration
```bash
# Install required packages
!pip install unsloth
!pip install transformers
!pip install accelerate
```

2. Unsloth Integration
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-8b-chat",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### Training Configuration
```python
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
)
```

## 🔄 RAG Implementation

Based on [Easy Local RAG](https://github.com/AllAboutAI-YT/easy-local-rag/blob/main/localrag.py)

### Components
1. Document Ingestion
```python
# Example document loading
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./medical_docs', glob="**/*.pdf")
documents = loader.load()
```

2. Vector Store Setup
```python
# ChromaDB initialization
from chromadb import Chroma
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

3. Query Pipeline
```python
def process_query(question):
    # Retrieve relevant documents
    docs = vectordb.similarity_search(question)
    
    # Generate context-aware response
    response = model.generate(
        context=docs,
        question=question
    )
    return response
```

## 🤖 AI Tools Integration

### Claude Integration
- Purpose: Data preprocessing and validation
- Implementation: API integration for data cleaning

### ChatGPT Integration
- Purpose: Response quality evaluation
- Implementation: Automated testing pipeline

## 📦 Model Deployment

### Hugging Face Hub Deployment
1. Model Packaging
```python
from transformers import AutoModelForCausalLM

# Save model
model.save_pretrained("medical-qa-model")
tokenizer.save_pretrained("medical-qa-model")
```

2. Upload to Hub
```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="medical-qa-model",
    repo_id="your-username/medical-qa-model",
    repo_type="model"
)
```

## 📊 Performance Metrics

1. Question-Answering Performance
- BLEU Score: [Your score]
- ROUGE Score: [Your score]
- F1 Score: [Your score]

2. Response Time
- Average latency: [Your latency]
- p95 latency: [Your p95]
- p99 latency: [Your p99]

## 🔧 Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd medical-qa-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model:
```bash
python download_model.py
```

## 🚀 Usage

### Local Deployment
```python
from medical_qa import MedicalQA

# Initialize the system
qa_system = MedicalQA()

# Ask a question
response = qa_system.ask("What are the symptoms of diabetes?")
print(response)
```

### API Endpoint
```bash
curl -X POST http://your-api-endpoint/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the symptoms of diabetes?"}'
```

## 📈 Future Improvements

1. Model Enhancements
- Implement multi-language support
- Add medical image understanding
- Improve context retention

2. Technical Optimizations
- Reduce model latency
- Implement model quantization
- Enhance RAG efficiency

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Meta AI Research Team for Llama 3.1
- Unsloth team for the optimization framework
- Kaggle for the medical dataset
- AllAboutAI-YT for the RAG implementation
