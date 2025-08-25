"""
=======================================
Customs function for sentiment analysis
=======================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 01-08-2025

Dette skript indeholder funktioner, som gør det muligt at hente
en sporgmodel fra HuggingFace, og anvende den til sentiment
analyse uden at det påkræver en forbindelse til nettet.
"""

# %% Setting things up

import re
import os
import pandas as pd
import nltk
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# %% Egne funktioner til beregning af sentiment score


def download_model(
    model_name: str = "alexandrainst/da-sentiment-base",
    save_directory: str = ".sentiment_model",
):
    """
    Downloads a Hugging Face model and tokenizer and saves them to a local directory.

    Args:
        model_name (str): the HuggingFace model ID. Defaults to
        'alexandrainst/da-sentiment-base')
        save_directory (str): the path to the local directory where
        the model should be saved. Defaults to ".sentiment_model".
    """
    # Ensuring we have all the necessary dependencies
    nltk.download("punkt")

    # Make sure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Save both to the specified directory
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to: {save_directory}")


def make_clf_pipeline(
    model_path=".sentiment_model",
) -> TextClassificationPipeline:
    """
    Initializes and returns a transformers model pipeline
    that can be used for text classification tasks such
    as sentiment analysis.

    Args:
        model_path (str, optional): Local path to the model directory.
        Defaults to ".sentiment_model".

    Returns:
        TextClassificationPipeline: CLF pipeline that can be
        used in the get_sentiment() and get_sentiment_df() functions.
    """

    # Check if model path exists and contains required files
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")

    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = [
        f for f in required_files if not os.path.isfile(os.path.join(model_path, f))
    ]

    if missing_files:
        raise FileNotFoundError(
            f"The following required files are missing in '{model_path}': {missing_files}"
        )

    # Load tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )

    # Initialize the pipeline
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

    print(f"Model and tokenizer successfully loaded from local path: '{model_path}'")

    return clf


def split_text_for_model(
    text: str,
    model_pipeline: TextClassificationPipeline,
    max_tokens: int = 512,
    buffer_tokens: int = 20,
    punctuation_buffer: bool = True,
) -> list[str]:
    """
    Splits a Danish-language text into chunks that fit within
    the language model's max token length. This is especially
    useful for longer texts as we often see in the FT data.
    The split is always done in between sentences, which helps
    preserve the original meaning of the text even when splitting.

    Args:
        text (str): The input text (can be long).
        model_pipeline (TextClassificationPipeline): the pipeline
        used for modelling. Needed to ensure that text is split
        using the same tokenizer employed by the language model.
        max_tokens (int): Max tokens the model can handle (default: 512).
        buffer_tokens (int): Safety margin to avoid token overflow.
        punctuation_buffer (bool): whether to add an additional space
        after punctuation marks used at the end of sentences (.!?).
        Helps ensure that sentences are split correctly even if the
        transcription of the text mistakenly omits the space after
        the punctuation mark. Defaults to True.

    Returns:
        list[str]: List of text chunks within the allowed token length.
    """
    # Ensuring sentences can be splti correctly
    if punctuation_buffer:
        text = re.sub(r"([.!?])(?!\s)", r"\1 ", text)

    # Initializing the sentence split
    tokenizer = model_pipeline.tokenizer  # use the correct tokenizer
    sentences = sent_tokenize(text, language="danish")

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        tokenized = tokenizer(sentence, add_special_tokens=False)
        token_count = len(tokenized["input_ids"])

        if current_tokens + token_count + buffer_tokens < max_tokens:
            current_chunk += " " + sentence
            current_tokens += token_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def get_sentiment(
    model_pipeline: TextClassificationPipeline,
    text: str,
    text_id: str | None = None,
) -> list[dict]:
    """
    Uses a pre-trained transformer model to automatically assign
    a given text to either an positive, negative or neutral
    sentiment category. Automatically splits longer texts into
    shorter texts to ensure we remain within the maximum token
    limit of the model.

    Args:
        model_pipeline (TextClassificationPipeline): CLF model pipeline
        to use for the language classification task.
        text (str): text for which to analyze
        text_id (str|None): an ID to add to the output, if needed.
        Defaults to None, which does not add an ID to the output.

    Returns:
        list[dict]: a list of one or more items (depending on
        the length of the original text and whether it needs
        splitting), where each item contains both the predicted
        sentiment and the model's confidence about the prediction
    """

    # Ensuring that longer texts are split to respect the model's
    # token limit
    text_for_model = split_text_for_model(text, model_pipeline)

    # Extracting sentiment for each part of the text & returning
    output = []
    for t in text_for_model:
        tmp_sentiment = model_pipeline(t)
        if text_id:
            tmp_sentiment[0]["id"] = text_id
        output.extend(tmp_sentiment)

    return output


def get_sentiment_df(
    model_pipeline: TextClassificationPipeline,
    texts: list[str],
    ids: list,
) -> pd.DataFrame:
    """
    A wrapper of the get_sentiment() function that returns output
    in a data frame format, where the sentiment score(s) for each
    text are also accompanied by their respective ID.
    """

    # We initialize a list to store the results
    all_sentiments = []

    # We loop over all data inputs to generate the scores
    for t, t_id in tqdm(zip(texts, ids), total=len(texts)):
        output_t = get_sentiment(model_pipeline, t, t_id)
        all_sentiments.extend(output_t)

    # We convert the output to a data frame format and return
    all_sentiments = pd.DataFrame(all_sentiments)
    return all_sentiments
