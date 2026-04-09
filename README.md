# Modernized NLP Project

This repository contains a collection of Natural Language Processing (NLP) projects, including a spam classifier, a TOEFL roleplay chatbot, and an exploration of Word2Vec and Average Word2Vec techniques. The original Jupyter notebooks have been refactored into a more organized project structure for clarity and ease of use.

## Features

-   **Spam Classifier**: A machine learning model to classify SMS messages as spam or ham using Bag-of-Words (BoW) and TF-IDF.
-   **TOEFL Roleplay Chatbot**: An interactive chatbot designed for TOEFL-style roleplay test sessions, providing structured grading using the Gemini API.
-   **Word2Vec Deep Dive**: An exploration of Word2Vec and Average Word2Vec for word embeddings and their applications in NLP tasks.

## Project Structure

The project is organized into the following directories:

```
modern_nlp_project/
├── data/
│   └── SMSSpamCollection.txt
├── notebooks/
│   ├── 1_NLP_Techniques_Experimentation.ipynb
│   ├── 2_TOEFL_Roleplay_Chatbot.ipynb
│   └── 3_Word2Vec_Deep_Dive.ipynb
├── src/
│   ├── 1_Spam_Classifier_BoW_TFIDF.py
│   ├── 2_TOEFL_Roleplay_Chatbot.py
│   └── 3_Word2Vec_Deep_Dive.py
├── .gitignore
└── requirements.txt
└── README.md
```

-   `data/`: Contains the datasets used by the projects.
-   `notebooks/`: Original Jupyter notebooks for reference and interactive exploration.
-   `src/`: Python scripts converted from the Jupyter notebooks, suitable for execution.
-   `.gitignore`: Specifies intentionally untracked files to ignore.
-   `requirements.txt`: Lists all Python dependencies required to run the projects.
-   `README.md`: This file, providing an overview and instructions.

## Setup

To set up the project environment, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/modern_nlp_project.git
    cd modern_nlp_project
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data**:

    Some notebooks/scripts require NLTK data (e.g., stopwords). Open a Python interpreter or add the following to a script and run it once:

    ```python
    import nltk
    nltk.download("stopwords")
    ```

5.  **Gemini API Key (for Chatbot)**:

    The `2_TOEFL_Roleplay_Chatbot.py` script (and its corresponding notebook) uses the Google Gemini API. You will need to set your `GEMINI_API_KEY` as an environment variable. Create a `.env` file in the root directory of the project with the following content:

    ```
    GEMINI_API_KEY="YOUR_API_KEY"
    ```
    Replace `YOUR_API_KEY` with your actual Gemini API key.

## Usage

### Running Python Scripts

You can run the Python scripts directly from the `src/` directory:

-   **Spam Classifier**: `python src/1_Spam_Classifier_BoW_TFIDF.py`
-   **TOEFL Roleplay Chatbot**: `python src/2_TOEFL_Roleplay_Chatbot.py`
-   **Word2Vec Deep Dive**: `python src/3_Word2Vec_Deep_Dive.py`

### Exploring Jupyter Notebooks

To interact with the original Jupyter notebooks, ensure you have Jupyter installed (`pip install jupyter`). Then, navigate to the `notebooks/` directory and start Jupyter Lab or Jupyter Notebook:

```bash
cd notebooks
jupyter lab
# or jupyter notebook
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: A LICENSE file is not included in this project, but it is good practice to add one.)

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
