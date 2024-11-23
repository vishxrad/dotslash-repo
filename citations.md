I'll help format this markdown content properly, ensuring consistent spacing and formatting:

# Citations and Acknowledgments

## Datasets

We utilized a modified version of the **"Comprehensive Medical Q&A Dataset"** originally published on Kaggle by *TheDevastator*. 

### Original Dataset

* **Source**: [Comprehensive Medical Q&A Dataset (Kaggle)](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset?resource=download)
* **Creator**: TheDevastator
* **Platform**: Kaggle

### Dataset Modifications

We adapted the original dataset through:

* Custom reformatting using JavaScript for our specific use case
* Upload to HuggingFace for improved ML pipeline integration
* Structured conversion to Alpaca-style prompts for instruction tuning

Modified Dataset Access: [visharxd/medical-qna on HuggingFace](https://huggingface.co/datasets/visharxd/medical-qna)

## Models and AI Technologies

We used an iteration of the colab notebook by Unsloth to fine tune our large language model: [Notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)

### Base Model

* **Meta-Llama-3.1-8B**: Used as our foundation model for medical query processing [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Training Techniques

We implemented the following training methodologies:

* Low-Rank Adaptation (LoRA) for efficient fine-tuning
* 4-bit quantization for reduced memory footprint
* Supervised Fine-Tuning (SFT) using the HuggingFace Trainer

### RAG

We used part of this notebook from AllAboutAI-YT to use ollama as a RAG model: [https://github.com/AllAboutAI-YT/easy-local-rag/blob/main/localrag.py](https://github.com/AllAboutAI-YT/easy-local-rag/blob/main/localrag.py)

## Technical Stack

### Frontend Technologies

* React (Latest version)
* Tailwind CSS for styling
* Vite for build tooling
* W3 and svgrepo SVG's
### Backend Technologies

* Django REST Framework
* PostgreSQL
* Docker for containerization

## Libraries and Frameworks

Our project builds upon these excellent open-source libraries:

### AI/ML Libraries

* `unsloth`: For efficient model loading and optimization
* `transformers`: HuggingFace's transformers library
* `datasets`: For dataset handling
* `trl`: For supervised fine-tuning

### Development Tools

* Node.js (v18.0.0+)
* Python (3.9+)
* Git for version control

## Development Assistance

* Claude AI (Anthropic) and ChatGPT (OpenAI): Used for debugging assistance during development

## Special Thanks

We extend our gratitude to:

* TheDevastator for creating and sharing the original medical Q&A dataset
* The HuggingFace team for their excellent model hosting and datasets platform
* The Tailwind CSS team for their utility-first CSS framework
* The React core team for their UI library
* The Django community for their robust backend framework
* The open-source community for their invaluable tools and libraries

## License

This project is built upon work licensed under various open-source licenses:

* React: MIT License
* Tailwind CSS: MIT License
* Django: BSD License
* Transformers: Apache License 2.0
* Original Dataset: Public Domain

Please refer to individual repositories and the original dataset page for detailed licensing information.

---

*Note: This citations document was prepared for Dot Slash Hackathon 2024. All trademarks and registered trademarks are the property of their respective owners.*
