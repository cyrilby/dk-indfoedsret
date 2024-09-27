"""
============================================
Enkel tekstanalyse af taler under debatterne
============================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 26-09-2024

Formålet ved dette skript er at tage de rensede data fra debatterne,
som indeholder diverse taler ("udtalelser"), og bruge dem til klassisk
tekstanalyse, herunder beregning af "sentiment score", identificering af
"n-grams", korrelationer, m.fl. Alt i den her skript foregår lokalt
uden at bruge eksterne API'er.

For the use of ChatGPT/Microsoft Copilot, which will be explored in a
separate script, visit this helpful tutorial:
https://www.browserbear.com/blog/using-chatgpt-api-in-python-5-examples/
"""

# %% Generel opsætning

# Import af relevante pakker
import pandas as pd
import sys
import subprocess
from collections import Counter
from tqdm import tqdm
from sentida import Sentida
import spacy

# Import af data, som allerede er blevet renset
all_debates = pd.read_parquet("output/ft_behandlinger.parquet")


# %% Egen funktion til beregning af sentiment score


def get_sentiment(input_text: str, fast: bool = False) -> list[float]:
    """
    Beregner gennemsnitlig og total 'sentiment score' for et
    stykke teksts, uanset tekstens længde.

    Args:
        input_text (str): tekst at beregne sentiment score på
        fast (bool): om hastighed skal prioriteres i beregningen.
        Pr. definition er hastighed ikke en prioritet.

    Returns:
        list[float]: gns. og total sentiment score
    """

    # Noterer brugerens valgte hastighed
    if fast:
        speed_type = "fast"
    else:
        speed_type = "normal"

    # Bererning af gennesmitlig score
    avg_score = Sentida().sentida(
        input_text,
        output="mean",
        normal=True,
        speed=speed_type,
    )

    # Bererning af total score
    total_score = Sentida().sentida(
        input_text,
        output="total",
        normal=True,
        speed=speed_type,
    )

    return [avg_score, total_score]


# %% Egen funktion til at sikre, at dansk sprogpakke er hentet


def download_spacy_model(model_name):
    """Function to download a spaCy model if it's not already installed."""
    try:
        # Check if the model is already installed
        spacy.load(model_name)
        print(f"Model '{model_name}' is already installed.")
    except OSError:
        print(f"Model '{model_name}' not found. Downloading now...")
        # Run the spacy download command
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
        print(f"Model '{model_name}' downloaded successfully!")


# %% Egen funktion til pre-processing af tekst


# Function to process a speech: tokenize, lemmatize, and filter stopwords/punctuation
def get_tokens(text: str, nlp: spacy.lang.da.Danish) -> list:
    """
    Tokenizes the text, excluding punctuation and stopwords.
    Assumes the Danish language pack has been installed.

    Args:
        text (str): text to preprocess.

    Returns:
        list: list of all tokens, including potential duplicates
    """
    #
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return tokens


# %% Sentiment analyse [WIP as of 26-09-2024]

"""
For hver eneste udtalelser beregner vi både gennemsnitlig og total
sentiment score. Den totale score muliggør det at lave efterfølgende
beregninger (fx i Power BI) med en højere præcisionsgrad. Vi bruger
pakken 'sentida', som er udviklet specifikt til at lave denne type
af analyse på det danske sprog.

Husk at give credit til pakkens udviklere fra Aarhus Universitet:
https://pypi.org/project/sentida/
"""

print("Beregning af sentiment score er nu i gang...")

# Vi starter med et nyt dataset
sentiment_scores = all_debates.copy()
all_speeches = sentiment_scores["Udtalelse"].tolist()

# Først laver vi beregningen, så noterer vi resultaterne i 2 særskilte kolonner
avg_sent_scores, total_sent_scores = [], []

for speech in tqdm(all_speeches, total=len(all_speeches)):
    avg_score, total_score = get_sentiment(speech)
    avg_sent_scores.append(avg_score)
    total_sent_scores.append(total_score)

sentiment_scores["GennemsnitligSentiment"] = avg_sent_scores
sentiment_scores["TotalSentiment"] = total_sent_scores

# Til sidst gemmer vi blot ID nummer og resultatkolonnerne
cols_to_keep = ["UdtalelseId", "GennemsnitligSentiment", "TotalSentiment"]
sentiment_scores = sentiment_scores[cols_to_keep]

print("Beregning af sentiment score færdig.")


# %% Beregning af ordfrekvenser m.fl. [WIP as of 26-09-2024]

# Vi henter den nødvendige sprogpakke hvis den ikke findes på PC'en
dk_lang_model = "da_core_news_sm"
download_spacy_model(dk_lang_model)

# Vi indlæser en personaliseret liste over stopwords, som er
# specifikke til indfødsretsdebatterne [WIP as of 26-09-2024]
# .... ==> Dette kunne også flyttes til Power BI - ved brug af
# mapping tabeller eller lignende

# Vi indlæser sprogpakken, som understøtter dansk tekst
nlp = spacy.load(dk_lang_model)

# Vi laver et nyt datasæt til at gemme resultaterne
cols_to_keep = ["UdtalelseId", "Udtalelse"]
speech_tokens = all_debates[cols_to_keep].copy()
all_speeches = speech_tokens["Udtalelse"].tolist()

# Vi laver pre-processing på alle udtalelser og noterer alle anvendte ord
all_tokens = []

for speech in tqdm(all_speeches, total=len(all_speeches)):
    temp_tokens = get_tokens(speech, nlp)
    all_tokens.append(temp_tokens)

# Vi tilføjer en ny række for hver eneste ord anvendt i hver eneste udtalelse
speech_tokens["Token"] = all_tokens
speech_tokens = speech_tokens.explode("Token")

# Til sidst gemmer vi blot ID nummer og resultatkolonnerne
cols_to_keep = ["UdtalelseId", "Token"]
speech_tokens = speech_tokens[cols_to_keep]


# %% Beregning af ordfrekvenser (skal ske i Power BI)

"""
============================
Kiril's comment, 26-09-2024:
============================
Below are some of ChatGPT's suggestions on how to work with this
in practice. I will modify it somewhat as I'd like to do the final
aggregations and data visualizations in Power BI, not Python.
"""

# Example text in Danish
text = "Dette er en eksempeltekst på dansk. Det danske sprog er et flot sprog."
text = all_debates["Udtalelse"].iloc[0]

# Tokenize and process the text
doc = nlp(text)

# Extract tokens while filtering out stopwords and punctuation, and lemmatizing the tokens
tokens = [
    token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct
]

# Word frequencies
word_freq = Counter(tokens)

# N-grams (bigram example)
bigrams = list(zip(tokens, tokens[1:]))

print(word_freq)
print(bigrams)


# %% Eksport af de færdige analyser

sentiment_scores.to_parquet("output/results_sentiment.parquet")
speech_tokens.to_parquet("output/results_speech_tokens.parquet")
print("Resultater fra tekstanalyse er eksporteret til 'output data' mappen.")
print("FÆRDIG.")


# %%
