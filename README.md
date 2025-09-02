# RAG Explorer for Wikipedia & Research Paper 

This repository demonstrates **Retrieval-Augmented Generation (RAG)** with two knowledge sources: **Wikipedia** and a **research paper about RAG**.  

## Features  
1. Choose `Wikipedia` as the knowledge source.  
2. Choose `Research Paper` as the knowledge source.  
3. Retrieve live and up-to-date information from Wikipedia.  
4. Retrieve focused insights about RAG from the research paper.  

## Setup

### Installation
1. Navigate to the project directory:
```bash
cd RAG_Explorer
```

2. Use a virtual environment 
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install requirements needed
```bash
pip install -r requirements.txt
```

4. Activate the virtual environment created by Poetry:
```bash
poetry shell
```

5. Install project dependencies using Poetry:
```bash
poetry install
```

6. Create a `.env` file and add your own OpenAI API key in the `.env` file as follows:
```
OPENAI_API_KEY=your-key-here
```

### Running the Application
```bash
streamlit run app.py
```
 Once the server starts, open a web browser and follow the link displayed by Streamlit to access the application.

### Usage
1. Upon launching the application, you'll be presented with a dropdown menu to select the information source: either `Wikipedia` or `Research Paper`.

2. Choose the desired source, and the app will retrieve relevant information based on your selection.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
