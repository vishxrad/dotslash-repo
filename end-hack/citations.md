# Citations and Acknowledgments

## Datasets

We utilized a modified version of the **"Comprehensive Medical Q&A Dataset"** originally published on Kaggle by *TheDevastator*. 

### Original Dataset

* **Source**: [Comprehensive Medical Q&A Dataset (Kaggle)](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset?resource=download)
* **Creator**: TheDevastator
* **Platform**: Kaggle

## Models and AI Technologies

We used an iteration of the colab notebook by Unsloth to fine tune our large language model: [Notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)

### Base Model

* **Meta-Llama-3.1-8B**: Used as our foundation model for medical query processing [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)


### RAG

We used part of this notebook from AllAboutAI-YT to use ollama as a RAG model: [https://github.com/AllAboutAI-YT/easy-local-rag/blob/main/localrag.py](https://github.com/AllAboutAI-YT/easy-local-rag/blob/main/localrag.py)


## Libraries and Frameworks

Our project builds upon these excellent open-source libraries:

### AI/ML Libraries

* `unsloth`: For efficient model loading and optimization
* `transformers`: HuggingFace's transformers library
* `datasets`: For dataset handling
* `trl`: For supervised fine-tuning
* `pytorch`: For finetuning the model and rag
* `ollama`: For running the rag
* `openai`: Used to communicate with local Ollama server
* `gradio`: building the front end and hosting the app locally
* `serpapi`: Interfaces with Google Search API, Used for finding doctors and medicines, Retrieves search results programmatically

## Development Assistance

* Claude AI (Anthropic) and ChatGPT (OpenAI): Used for debugging assistance during development

## API used:
* SerpAPI: https://serpapi.com/

## Special Thanks

We extend our gratitude to:

* TheDevastator for sharing the original medical Q&A dataset
* The HuggingFace team for their excellent model hosting and datasets platform
* The open-source community for their invaluable tools and libraries

## License

This project is built upon work licensed under various open-source licenses:
* Llama 3.1: COMMUNITY LICENSE
* Ollama: MIT License
* Transformers: Apache License 2.0
* Original Dataset: Public Domain

Please refer to individual repositories and the original dataset page for detailed licensing information.

---

*Note: This citations document was prepared for Dot Slash Hackathon 2024. All trademarks and registered trademarks are the property of their respective owners.*
