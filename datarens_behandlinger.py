"""
====================================================
Import og rens af data om behnadlinger af lovforslag
====================================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 18-12-2024

Formålet ved dette skript er at indlæse data om de folketings-
debatter, som finder sted ved 1., 2. og 3. behandling i salen,
fra diverse HTM(L)-filer, samt strukturere de tekstbaserede data
i et format, som senere kan bruges til dataanalayse og/eller
visualisering.
"""

# %% Generel opsætning

# Import af relevante pakker
import pandas as pd
import numpy as np
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

# Skal alle HTML filer genindlæses eller skal kun de nye indlæses
read_all_files = False

# Mappe, som indeholder HTM(L) filerne, der skal indlæses
import_folder = "input/debatter/"
import_files = [
    f
    for f in os.listdir(import_folder)
    if os.path.isfile(os.path.join(import_folder, f))
]

# Import af mappingtabel med partigrupper og roller
mapping_parties = pd.read_excel(
    "input/mapping_tabeller.xlsx", sheet_name="Partigrupper"
)

# Allerede rensede data på behandlinger
all_debates_prev = pd.read_parquet("output/ft_behandlinger.parquet")
debate_duration_prev = pd.read_parquet("output/ft_debatlængde.parquet")

# Vi tjekker hvilke filer, der tidligere er blevet bearbejdet
prev_files = all_debates_prev["Kilde"].unique().tolist()
new_files = [file for file in import_files if file not in prev_files]


# %% Egen funtion, som importerer en lokal HTML fil og omdanner den til BS4 objekt


def html_to_bs4(lokal_fil: str) -> BeautifulSoup:
    """
    Indlæser en lokal HTM(L) fil, og omdanner den til et BS4 objekt.

    Args:
        lokal_fil (str): lokal fil til indlæsning.

    Returns:
        BeautifulSoup: BS4 objekt til videre brug.
    """

    # Indlæsning af HTML indhold og konvertering til BS4-objekt
    with open(lokal_fil, "r", encoding="utf-8") as file:
        html_content = file.read()
    html_content = BeautifulSoup(html_content, "html.parser")
    return html_content


# %% Egen funktion, som finder datoen for mødet


def extract_date(html_content: BeautifulSoup) -> str:
    """
    Finder ud af, hvilke dato et folketingsmøde har fundet sted på,
    og leverer den tilbage formatteret som tekst.

    Args:
        html_content (BeautifulSoup): HTML side, som dækker over
        hele mødets indhold.

    Returns:
        str: mødedato i format "".
    """

    # Vi kan finde datoen i overskriften af HTML-siden
    meeting_date = html_content.find_all(
        "h2", class_="TingDokDocumentHeaderSpotA__heading"
    )[0].text
    meeting_date = meeting_date[-10:]
    return meeting_date


# %% Egen funktion, som identificerer forskellige typer af udtalelser


def extract_types(html_content: BeautifulSoup) -> list:
    """
    Evaluerer indholdet af en HTML side for at finde ud af, hvilke
    "div class" typer skal tælles med som udtalelser under debatten.

    Args:
        html_content (BeautifulSoup): BS4 objekt, som dække over
        indholdet af hele HTML-siden.

    Returns:
        list: liste over alle relevante "div class" typer.
    """

    # Vi starter med at kigge på de forskellige type af udtalelser, som vi identificerer
    # automatisk pba. at de skal indeholde både "negotiation__item" og "speakType"
    divs_with_negotiation_item = html_content.find_all(
        "div", class_=re.compile(r"negotiation__item.*speakType")
    )
    allowed_classes = set()
    for div in divs_with_negotiation_item:
        allowed_classes.add(" ".join(div["class"]))
    allowed_classes = list(allowed_classes)
    allowed_classes.sort()
    return allowed_classes


# %% Egen funktion, som omdanner en enkel udtalelse til df


def structure_statement(speech_entry: BeautifulSoup) -> pd.DataFrame:
    """
    Bruger HTML koden, som tilhører til den enkelte udtalelse,
    til at udtage nogle bestemte informationer om talen samt
    talens indhold, og producerer en df pba. disse oplysninger.

    Args:
        speech_entry (BeautifulSoup): BS4 objekt, som indeholder en
        enkelt udtalelse

    Returns:
        pd.DataFrame: struktureret tale i df format
    """
    speech_time = speech_entry.find("div", class_="negotiation__time")
    speech_position = speech_time.find_next_sibling("div").text
    speech_time = speech_time.text
    speech_name = speech_entry.find("div", class_="negotiation__name").text
    # Note: der er nogle forskelle på hvilken class indeholder udtalelsen
    speech_text = speech_entry.find(
        "p", class_=lambda class_name: class_name in ["Tekst", "TekstIndryk"]
    )
    if speech_text:
        speech_text = speech_text.text
    speech_entry = pd.DataFrame(
        {
            "Tid": [speech_time],
            "Navn": [speech_name],
            "Rolle": [speech_position],
            "Udtalelse": [speech_text],
        }
    )
    return speech_entry


# %% Egen funktion, som omdanner en hel HTML side med utalelser til df


def extract_statements(lokal_fil: str) -> pd.DataFrame:
    """
    Går igennem HTML-koden, som indeholder alle udtalelser, som
    stammer fra en lokal HTM(L) fil, og strukturerer hver eneste
    udtalelse i et df-format. Leverer en df, hvor de strukturerede
    oplysninger fremstår klar til brug i videre analyser.

    Args:
        lokal_fil (str): lokal HTM(L) fil til indlæsning (typisk en hel
        side hentet fra www.ft.dk)

    Returns:
        pd.DataFrame: df, som indeholder alle udtalelser fra HTML-siden
    """

    # Allerførst skal vi indlæse HTML filen som et BS4 objekt
    html_content = html_to_bs4(lokal_fil)

    # Først finder vi ud af, hvornår mødet har fundet sted
    meeting_date = extract_date(html_content)

    # Bagefter identificerer vi alle relevante "div class" typer
    allowed_classes = extract_types(html_content)

    # Nu finder vi alle DIVs, og uddrager oplysninger hvis der findes nogle
    div_elements = html_content.find_all("div")
    all_speeches = []

    # Nu går vi igennem alle "div" elementer og tjekker om de er i "allowed_classes"
    for div in div_elements:
        # Først kigger på "class" først (som også kan være "None" nogle gange)
        div_class = div.get("class")
        if div_class:
            div_class = " ".join(div_class)

        # Så kigger vi på om "class" falder inden for en af de "allowed_classes"
        if div_class in allowed_classes:
            # print(f"Div with class {div_class} is allowed.")
            temp_speech = structure_statement(div)
            all_speeches.append(temp_speech)

    # Som næste skridt sætter vi dataene sammen, tilføjer timestamp osv.
    all_speeches = pd.concat(all_speeches)
    all_speeches = all_speeches.reset_index(drop=True)
    all_speeches = all_speeches[all_speeches["Navn"].str.len() > 1]
    all_speeches["Tid"] = meeting_date + " " + all_speeches["Tid"]
    all_speeches["Tid"] = all_speeches["Tid"].str.replace(".", ":", regex=False)
    # OBS: I de ældre HTML-filer er tid ikke angivet for alle undtalelser
    all_speeches["Tid"] = np.where(
        all_speeches["Tid"].str.len() > 11, all_speeches["Tid"], np.nan
    )
    all_speeches["Tid"] = pd.to_datetime(all_speeches["Tid"], format="%d-%m-%Y %H:%M")

    # Til sidst fjerner vi eventuelle dubletter
    id_cols = ["Tid", "Navn", "Udtalelse"]
    all_speeches = all_speeches.drop_duplicates(subset=id_cols)
    all_speeches = all_speeches.reset_index(drop=True)
    return all_speeches


# %% Egen funktion, som sammensætter alle udtalelser fra forskellige debatter


def combine_all_statements(html_files: list[str], folder: str = None) -> pd.DataFrame:
    """
    Går igennem en liste af lokale HTML-filer, indlæser hver eneste
    fil og identificerer debatdato samt strukturerer alle udtalelser,
    tilgængelige i den enkelte fil. Til sidst sammensætter udtaleleser
    fra alle HTML-filer i en df, og tilføjer en unik ID nummer for
    hver enkel udtalelse.

    Args:
        html_files (list[str]): liste over relevante HTML-filer, som
        skal indlæses og hvis udtalelser skal struktutereres i en df.
        folder (str): mappe, som indeholder de individuelle HTML filer
        (hvis relevant).

    Returns:
        pd.DataFrame: alle registrerede udtalelser fra HTML-filerne
        i et df-format.
    """

    # Først indlæser vi og sammensætter alle HTML-filer
    all_content = []
    for html_page in tqdm(html_files, total=len(html_files)):
        if folder:
            file_to_import = folder + html_page
        else:
            file_to_import = html_page
        page_content = extract_statements(file_to_import)
        page_content["Kilde"] = html_page
        all_content.append(page_content)

    # Bagefter konverterer vi til df format
    all_content = pd.concat(all_content)
    all_content = all_content.reset_index(drop=True)

    # Til sidst tilføjer vi info om "År" og "Sæson", som lovforslaget vedrører,
    # og bruge disse informationer til at danne et unikt ID for hver udtalelse
    all_content["År"] = all_content["Kilde"].str.split("_").str[0]
    all_content["Sæson"] = all_content["Kilde"].str.split("_").str[1]
    all_content["BehandlingNr"] = all_content["Kilde"].str.split("_").str[2]
    all_content["År"] = all_content["År"].astype(int)
    all_content["Sæson"] = all_content["Sæson"].str.title()
    all_content["BehandlingNr"] = all_content["BehandlingNr"].astype(str)
    return all_content


# %% Egen funktion til beregning af antal ord i en tekst


def count_words(text: str) -> int:
    """
    Beregner antallet af ord i en tekts.
    Der skal være mindst et mellemrum mellem 2 ord for
    at de kan blive optalt som 2 selvstændige ord.

    Args:
        text (str): _description_

    Returns:
        int: _description_
    """

    # Split by any non-word characters (like punctuation) and
    # filter out empty strings, then count the N of words
    n_words = len(re.findall(r"\b\w+\b", text))
    return n_words


# %% Import og rens af data

print("Import og rens af data er nu i gang...")

# Vi indlæser enten alle eller kun de nye HTML filer
if not read_all_files:
    import_files = new_files

print(f"Bemærk: {len(import_files)} fil(er) med debatsdata vil blive indlæst.")

# Import af alle filers indhold
all_debates = combine_all_statements(import_files, import_folder)

# Eftersom ikke alle tider er angivet i de allerældste data er vi
# nødt til at bruge en form for afrunding for de manglende tider
all_debates["Tid"] = all_debates.groupby("Kilde")["Tid"].bfill()
all_debates["Tid"] = all_debates.groupby("Kilde")["Tid"].ffill()

# Derudover kan der nogle gange opstå intervaller i start/slut af teksterne,
# så disse bliver fjernet i alle relevante kolonner
relevant_cols = ["Navn", "Rolle", "Udtalelse"]
for col in relevant_cols:
    all_debates[col] = all_debates[col].str.strip()

# Endvidere er ens navn og rolle nogle gange også skrevet i starten af udtalelsen, så vi retter også på det
all_debates["TekstForTjek"] = all_debates["Udtalelse"].str.split(":").str[0]
all_debates["NavnTjek"] = all_debates.apply(
    lambda row: row["Navn"] in row["TekstForTjek"], axis=1
)
all_debates["RolleTjek"] = all_debates.apply(
    lambda row: row["Rolle"] in row["TekstForTjek"], axis=1
)
all_debates["UdtalelseBegynder"] = all_debates["TekstForTjek"].str.len()
all_debates["UdtalelseBegynder"] = all_debates["UdtalelseBegynder"] + 2
all_debates["UdtalelseBegynder"] = np.where(
    all_debates["NavnTjek"] & all_debates["RolleTjek"],
    all_debates["UdtalelseBegynder"],
    0,
)
all_debates["UdtalelseSlutter"] = all_debates["Udtalelse"].str.len()
all_debates["Udtalelse"] = all_debates.apply(
    lambda row: row["Udtalelse"][row["UdtalelseBegynder"] : row["UdtalelseSlutter"]],
    axis=1,
)
cols_for_check = [
    "TekstForTjek",
    "NavnTjek",
    "RolleTjek",
    "UdtalelseBegynder",
    "UdtalelseSlutter",
]
all_debates = all_debates.drop(columns=cols_for_check)

# Nogle roller er stavet udelukkende med store bogstaver, det retter vi på her
all_debates["Rolle"] = all_debates["Rolle"].str.title()

# Dem, der mangler rolle, betragtes som almindelige folketingsmedlemmer
all_debates["Rolle"] = np.where(
    all_debates["Rolle"] != "", all_debates["Rolle"], "Alm. Folketingsmedlem"
)

# Til sidst tilføjer vi et unikt nummer for alle udtalelser
# Nummeret formatteres som "År-Sæson-BehandlingNr-UdtalelseNr"
id_cols = ["År", "Sæson", "BehandlingNr"]
all_debates["UdtalelseNr"] = all_debates.groupby(id_cols).cumcount() + 1
speech_max_digits = np.max(all_debates["UdtalelseNr"].astype(str).str.len())
all_debates["UdtalelseId"] = (
    all_debates["UdtalelseNr"].astype(str).str.zfill(speech_max_digits)
)
all_debates["SæsonId"] = np.where(all_debates["Sæson"] == "Forår", "F", "E")
all_debates["UdtalelseId"] = (
    all_debates["År"].astype(str)
    + "-"
    + all_debates["SæsonId"]
    + "-"
    + all_debates["BehandlingNr"].astype(str)
    + "-"
    + all_debates["UdtalelseId"]
)
cols_for_id = ["SæsonId"]
all_debates = all_debates.drop(columns=cols_for_id)

print(f"Bemærk: {len(all_debates)} rækker af data indsamlet er renset.")


# %% Markering af partigruppe

# Kolonnen "Navn" indeholder en forkortelse for partigruppe
all_debates["PartiGruppeKort"] = all_debates["Navn"].str.extract(r"\((.*?)\)")

# Der er deltagere i debatten, som ikke hører til en partigruppe
# som fx Folketingets formand, udlændingeministeren osv.
all_debates["PartiGruppeKort"] = all_debates["PartiGruppeKort"].fillna(
    all_debates["Rolle"]
)

# Vi tilføjer partigruppens fulde navn ved brug af vores egen mapping
all_debates = pd.merge(all_debates, mapping_parties, how="left", on="PartiGruppeKort")

print("Markering af partigruppe færdig.")


# %% Beregning af debatlængde

# Start og sluttidspunkt for debat
id_cols = ["År", "Sæson", "BehandlingNr"]
cols_to_keep = ["DebatStartTid", "DebatSlutTid"]
debate_duration = all_debates.copy()
debate_duration["DebatStartTid"] = debate_duration.groupby(id_cols)["Tid"].transform(
    "min"
)
debate_duration["DebatSlutTid"] = debate_duration.groupby(id_cols)["Tid"].transform(
    "max"
)
debate_duration = debate_duration.drop_duplicates(subset=id_cols)
debate_duration = debate_duration.reset_index(drop=True)
debate_duration = debate_duration[id_cols + cols_to_keep]

# Debattens længde beregnes i antal minutter
debate_duration["DebatLængdeMinutter"] = (
    debate_duration["DebatSlutTid"] - debate_duration["DebatStartTid"]
).dt.total_seconds() / 60

print("Beregning af debatlængde færdig.")


# %% Beregning af udtalelseslængde

# Først beregner vi udtalelseslængde i tid (antal minutter)
id_cols = ["År", "Sæson", "BehandlingNr"]
all_debates["NæsteTaleBegynder"] = all_debates.groupby(id_cols)["Tid"].shift(-1)
all_debates["UdtalelseLængdeMinutter"] = (
    all_debates["NæsteTaleBegynder"] - all_debates["Tid"]
).dt.total_seconds() / 60
all_debates = all_debates.drop(columns="NæsteTaleBegynder")

# Hvis en tale varer 0 eller NAN minutter, så er det fordi det er
# en meget kort tale - vi markerer den derfor med 1 minuts længde
all_debates["UdtalelseLængdeMinutter"] = np.where(
    (all_debates["UdtalelseLængdeMinutter"] == 0)
    | (all_debates["UdtalelseLængdeMinutter"].isna()),
    1,
    all_debates["UdtalelseLængdeMinutter"],
)

# Efter det beregner vi udtalelseslængde i antal ord
all_debates["UdtalelseLængdeOrd"] = all_debates["Udtalelse"].apply(count_words)

print("Beregning af udtalelselængde færdig.")


# %% Eksport af de rensede data

# Hvis vi kun har indlæst nye HTML filer, så skal vi sikre, at
# de datasæt, der eksporteres, også indeholder data fra de ældre
# HTML filer
if not read_all_files:
    all_debates = pd.concat([all_debates_prev, all_debates])
    debate_duration = pd.concat([debate_duration_prev, debate_duration])

all_debates.to_parquet("output/ft_behandlinger.parquet")
debate_duration.to_parquet("output/ft_debatlængde.parquet")
print("Rensede data på debatter er nu klar til brug i 'output' mappen.")
print("FÆRDIG.")

# %%
