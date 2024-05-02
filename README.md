# Repository Overview

This repository contains various tasks and experiments related to natural language processing, focusing on sentence similarity, phrase analysis, and word similarity scoring. The repository is organized into three main folders, each dedicated to a specific aspect of NLP.

## Folder Structure

### 1. Bonus Task

This folder includes notebooks for fine-tuning and inference experiments with various large language models (LLMs).

- **Fine-tune.ipynb**: This notebook contains the script for fine-tuning the BERT model on a specific dataset to adapt it for more specialized tasks.
- **Prompt_2.ipynb**: This notebook demonstrates the use of zero-shot and few-shot learning techniques with multiple LLMs to perform inference without extensive training.

### 2. Phrase and Sentences

This folder focuses on utilizing different embedding models to analyze phrases and sentences.

- **Glovee.ipynb**: Utilizes the GloVe pre-trained model to generate word embeddings that capture semantic meanings.
- **Phrase.ipynb**: Converts embeddings from the GloVe model into the Word2Vec format for compatibility with different analysis tools.
- **word2vec_sentences.ipynb**: Employs a pre-trained Word2Vec model to generate text embeddings, facilitating various NLP tasks.
- **Scratch_working.ipynb**: Develops a Word2Vec model from scratch, training it on a custom dataset to better understand the nuances of embedding training.

### 3. Word Similarity Score

This folder contains notebooks aimed at generating and analyzing word similarity scores using different datasets and pre-trained models.

- **Pre-trained-glove.ipynb**: Generates vectors using a pre-trained GloVe model, which are then used for calculating word similarity.
- **word_similarity_cellphone.ipynb**: Applies a Word2Vec model trained on a cellphone dataset to compute word similarities within the tech domain.
- **word_similarity_sports.ipynb**: Uses a Word2Vec model trained on sports-related data to explore word similarities in sports contexts.
- **Experiments**: A sub-folder containing notebooks that document various failed experiments related to word similarity calculations, providing insights into the challenges and pitfalls encountered during the experimentation phase.

