"""
========================================
Identificering af talestemning/sentiment
========================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 01-08-2025

Formålet ved dette skript er at anvende en sprogmodel for at
klassificere samtlige udtalelser i salen under enten positiv, negativ
eller neutral stemning (sentiment). Dette gøres ved brug af en lokal
model (uden eksterne API'er), oprindeligt hentet fra HuggingFace.
"""

# %% Generel opsætning

# Import af relevante pakker
import os
import pandas as pd
from functions_sentiment import download_model, make_clf_pipeline, get_sentiment_df

# Tving genberegning af sentiment score selv om der ikke er nye data
force_sentiment = True  # WIP as of 01-08-2025

# Tving offline kørsel af HuggingFace modellen
download_model()
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Import af data, som allerede er blevet renset
all_debates = pd.read_parquet("output/ft_behandlinger.parquet")

# Import af tidligere beregninger af sentiment
file_path_sentiment = "output/results_sentiment.parquet"
prev_sentiment = (
    pd.read_parquet(file_path_sentiment)
    if os.path.exists(file_path_sentiment)
    else None
)


# %% Sentiment analyse

"""
For hver eneste udtalelser beregner vi både gennemsnitlig og total
sentiment score. Den totale score muliggør det at lave efterfølgende
beregninger (fx i Power BI) med en højere præcisionsgrad. Vi bruger
en model fra HuggingFace, som er udviklet specifikt til at lave denne
type af analyse (tekstklassificering) på det danske sprog.

For mere info om modellen, besøg dens hjemmeside:
https://huggingface.co/alexandrainst/da-sentiment-base
"""

print("Beregning af sentiment score er nu i gang...")

# Vi beregner kun sentiment for nye udtalelser
if force_sentiment:
    new_speeches = all_debates.copy()
elif prev_sentiment is not None and not prev_sentiment.empty:
    new_speeches = all_debates[
        ~all_debates["UdtalelseId"].isin(prev_sentiment["UdtalelseId"])
    ].copy()
else:
    new_speeches = all_debates.copy()
n_new_speeches = len(new_speeches)

if n_new_speeches:
    print(f"Obs: {n_new_speeches} nye udtalelser vil blive analyseret.")

    # Vi starter en pipeline til sprogklassificering
    model_pipeline = make_clf_pipeline()

    # Vi forbereder data til brug i vores pipeline
    all_speeches = new_speeches["Udtalelse"].tolist()
    all_ids = new_speeches["UdtalelseId"].tolist()

    # Vi beregner sentiment score for alle relevante udtalelser
    sentiment = get_sentiment_df(model_pipeline, all_speeches, all_ids)

    # Vi oversætter outputtet til dansk
    col_names = {"label": "Stemning", "score": "Sandsynlighed", "id": "UdtalelseId"}
    translations = {"negative": "Negativ", "positive": "Positiv", "neutral": "Neutral"}
    sentiment = sentiment.rename(columns=col_names)
    sentiment["Stemning"] = sentiment["Stemning"].replace(translations)

    # Vi tilføjer ID'er i tilfælde af, at en udtalelses stemning
    # er delt over flere rækker pga. at udtalelsen er for lang
    sentiment["UdtalelseDel"] = sentiment.groupby("UdtalelseId").cumcount() + 1

    # Vi ændrer på kolonnernes orden
    cols_to_keep = [
        "UdtalelseId",
        "UdtalelseDel",
        "Stemning",
        "Sandsynlighed",
    ]
    sentiment = sentiment[cols_to_keep]

    # Vi sammensætter data med tidligere beregninger
    sort_cols = ["UdtalelseId", "UdtalelseDel"]
    sentiment = pd.concat([prev_sentiment, sentiment])
    sentiment = sentiment.sort_values(sort_cols)
    sentiment = sentiment.reset_index(drop=True)

    # Eksport af den færdige analyse
    sentiment.to_parquet("output/results_sentiment.parquet")
    print("Beregning af sentiment score færdig.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# !!!!!!!!!!!!!!!!!!!! WIP !!!!!!!!!!!!!!!!!!!!
# Remember to add an integer column for sentiment (-1, 0 or +1)
# as we would need that for Power BI visualizations...

print("FÆRDIG.")

# %%
