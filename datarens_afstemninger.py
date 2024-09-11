"""
======================================
Import og rens af data om afstemninger
======================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 11-09-2024

Formålet ved dette skript er at indlæse data om folketingsafstemninger
fra HTM(L)-filer, finde den relevante tabel med de personlige stemmer,
samt kombinere alle relevante tabeller i et samlet datasæt, som kan
bruges til dataanalayse og/eller visualisering.
"""

# %% Generel opsætning

# Import af relevante pakker
import pandas as pd
import os
from tqdm import tqdm

# Mappe, som indeholder HTM(L) filerne, der skal indlæses
import_folder = "input/afstemninger/"
files = [
    f
    for f in os.listdir(import_folder)
    if os.path.isfile(os.path.join(import_folder, f))
]
files = [f for f in files if "htm" in f]


# %% Import og rens af data

print("Import og rens af data er nu i gang...")
print(f"Bemærk: {len(files)} fil(er) med afstemningsdata vil blive indlæst.")

# Tomt objekt, som vil blive udfyldt med data
final_votes = []

# Import af hver eneste fil til pandas df
# Bemærk: det er altid den sidste tabel, som indeholder de individuelle stemmer
for file in tqdm(files, total=len(files)):
    temp_votes = pd.read_html(import_folder + file)[-1]
    temp_votes["Kilde"] = file
    final_votes.append(temp_votes)

# Samling af data i et datasæt
final_votes = pd.concat(final_votes)
final_votes = final_votes.sort_values("Kilde")
final_votes = final_votes.reset_index(drop=True)
final_votes["Kilde"].value_counts()

# Videre rensning
final_votes["År"] = final_votes["Kilde"].str.slice(0, 4).astype(int)
final_votes["Sæson"] = final_votes["Kilde"].str.split("_").str[1].str.title()
cols_order = ["År", "Sæson", "Navn", "Partigruppe", "Stemme"]
final_votes = final_votes[cols_order]
print(f"Bemærk: {len(final_votes)} rækker af data indsamlet er renset.")


# %% Eksport af de rensede data

final_votes.to_parquet("output/ft_afstemninger.parquet")
print("Rensede data på afsteminger er nu klar til brug i 'output' mappen.")
print("FÆRDIG.")

# %%
