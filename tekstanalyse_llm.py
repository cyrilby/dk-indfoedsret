"""
====================================
Tekstanalyse ved brug af OpenAIs LLM
====================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 26-11-2024

I dette skript anvender vi OpenAIs sprogmodel til forskellige slags
foremål, primært opsummering af lange udtalelser og gruppering af
udtalelser i nogle kategorier (emner). Pointen ved denne analyse er
at supplere de andre emneanalyser, som bruger klassiske værktøjer,
og skabe nogle indsigter, som er mere letforståelige end fx LDA.
"""

# %% Generel opsætning

# Import af relevante pakker
import os
import pandas as pd
import numpy as np
from functions_llm import connect_to_openai, get_prompt, query_llm_multiple

# Om visse dele af analysen skal opsummeres
# (Gælder kun i tilfælde, hvor ny data ikke trackes automatisk)
# OBS: Sat til "True" hvis vi har nye input data
summarize_party_opinions = False
summarize_personal_opinions = False

# Maks antal requests pr. minut for Azure OpenAI API
max_rpm = None  # or 20

# Maks længde for tekster, som ikke skal opsummmeres (antal tegn)
max_length = 160

# Import af renset input data
all_debates = pd.read_parquet("output/ft_behandlinger.parquet")
speech_tokens = pd.read_parquet("output/results_speech_tokens.parquet")

# Import af resultater fra den klassiske tekstanalyse
auto_topics = pd.read_parquet("output/results_topics_auto.parquet")
auto_words = pd.read_parquet("output/results_words_auto.parquet")

# Import af tidligere opsummeringer m.fl.
file_path_speech_summaries = "output/results_llm_opsummering.parquet"
prev_speech_summaries = (
    pd.read_parquet(file_path_speech_summaries)
    if os.path.exists(file_path_speech_summaries)
    else None
)
file_path_topics = "output/results_llm_lda_topics.parquet"
prev_topic_auto_names = (
    pd.read_parquet(file_path_topics) if os.path.exists(file_path_topics) else None
)

# Forbereder endpoint og headers til Azure OpenAI API
api_access = "credentials/azure_openai_access.yaml"
openai_client = connect_to_openai(api_access)


# %% Forberedelse af data til brug ifm. OpenAI model

# Vi tilføjer en kolonne, som viser lovforslagssæson
all_debates["Sæson"] = all_debates["UdtalelseId"].str.split("-").str[:2].str.join("-")

# Vi markerer rækker med ikke-polistiske udtalelser
condition_1 = all_debates["PartiGruppe"] == "Folketingets formand"
condition_2 = all_debates["Rolle"].str.lower().str.contains("formand")
all_debates["ApolitiskUdtalelse"] = (condition_1) | (condition_2)

# Vi markerer de resterende rækker, som skal bruges ifm. OpenAI modellen
all_debates["OpsummerUdtalelse"] = ~all_debates["ApolitiskUdtalelse"]


# %% Opsummering af alle relevante udtalelser

"""
Vi opsummerer alle politiske udtalelser, som er længere end en
sætning. Dvs. at Folketingets formands udtalelser samt meget korte
udtalelser vil ikke blive forkortet da de allerede er ret korte.
"""

print("Opsummering af alle relevante udtalelser er nu i gang...")

# Vi giver modellen følgende instruktioner
system_prompt = get_prompt("opsummering.txt")

# Vi opsummerer kun data, som er nye
if prev_speech_summaries is not None and not prev_speech_summaries.empty:
    new_speeches = all_debates[
        ~all_debates["UdtalelseId"].isin(prev_speech_summaries["UdtalelseId"])
    ].copy()
else:
    new_speeches = all_debates.copy()
n_new_speeches = len(new_speeches)

if n_new_speeches:
    # Vi opsummerer kun længere og politiske udtalelser
    to_summarize = new_speeches[new_speeches["OpsummerUdtalelse"]].copy()
    not_to_summarize = new_speeches[~new_speeches["OpsummerUdtalelse"]].copy()
    full_speeches = to_summarize["Udtalelse"].tolist()
    full_ids = to_summarize["UdtalelseId"].tolist()

    # Vi opsummerer alle relevante udtalelser
    tmp_summary = query_llm_multiple(
        openai_client, system_prompt, full_speeches, full_ids, max_rpm=max_rpm
    )

    # Vi omdøber visser kolonner
    col_names = {
        "Id": "UdtalelseId",
        "Response": "Opsummering",
        "hate": "SprogbrugHad",
        "self_harm": "SprogbrugSelvskade",
        "sexual": "SprogbrugSex",
        "violence": "SprogbrugVold",
    }
    tmp_summary = tmp_summary.rename(columns=col_names)

    # Vi tilføjer opsummeringerne til datasættet og fjerner andre kolonner
    tmp_summary["OpsummeretAfModel"] = "GPT-4o-mini"

    # Vil tilføjer de oprindelige taler til datasættet
    cols_to_keep = ["UdtalelseId", "Opsummering"]
    not_to_summarize["Opsummering"] = not_to_summarize["Udtalelse"]
    not_to_summarize = not_to_summarize[cols_to_keep].copy()
    not_to_summarize["OpsummeretAfModel"] = "Nej"

    # Vi kombinerer både opsummerede og ikke-opsummerede udtalelser
    speech_summaries = pd.concat([tmp_summary, not_to_summarize])
    speech_summaries = speech_summaries.sort_values("UdtalelseId")
    speech_summaries = speech_summaries.reset_index(drop=True)

    # Vi sammensætter tidligere og nuværende resultater i et datasæt
    speech_summaries = pd.concat([prev_speech_summaries, speech_summaries])

    # Vi bruger vores egen mapping til "Sprogbrug" kolonnerne
    lang_mapping = {
        "safe": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        None: np.nan,
        "Unknown": np.nan,
    }
    cols_to_map = [
        "SprogbrugHad",
        "SprogbrugSelvskade",
        "SprogbrugSex",
        "SprogbrugVold",
    ]
    for col in cols_to_map:
        speech_summaries[col] = speech_summaries[col].map(lang_mapping)

    # Vi sorterer og eksporterer data
    speech_summaries = speech_summaries.sort_values("UdtalelseId")
    speech_summaries = speech_summaries.reset_index(drop=True)
    speech_summaries.to_parquet("output/results_llm_opsummering.parquet")

    print("Automatisk opsummering af udtalelser færdig.")

else:
    speech_summaries = prev_speech_summaries.copy()
    print("Obs: Springer over pga. mangel af nye input data.")


# %% LDA emner til menneskesprog

"""
Vi forsøger at danne menneskevenlige navne på de emner, som vores LDA
model har skabt. Dette sker ved at udvalge de top 20% bedst klassificerede
udtalelser, og skabe en samlet tekst af deres opsummeringer. Vi beder derefter
OpenAI modellen til at lave en kort, men deskriptiv overskrift til hver tekst.
"""

print("Omdannelse af LDA emner til menneskesprog er nu ingang...")

# For hvert emne finder vi de top 20% udtalelser med højest præcision
pct_for_sampling = 0.2
sort_vars = ["Sæson", "EmneNr", "Sandsynlighed"]
sort_vars_asc = [True, True, False]
group_vars = ["Sæson", "EmneNr"]
best_fitting = auto_topics.copy()
best_fitting = best_fitting.sort_values(by=sort_vars, ascending=sort_vars_asc)
best_fitting["ScoreRank"] = best_fitting.groupby(group_vars).cumcount() + 1
best_fitting["MaxRank"] = best_fitting.groupby(group_vars)["ScoreRank"].transform("max")
best_fitting["ScoreRank_Pct"] = best_fitting["ScoreRank"] / best_fitting["MaxRank"]

# Vi beriger dataene med opsummeringen af de enkelte udtalelser
best_fitting = best_fitting[best_fitting["ScoreRank_Pct"] <= pct_for_sampling].copy()
topic_summaries = pd.merge(
    best_fitting,
    speech_summaries[["UdtalelseId", "Opsummering"]],
    how="left",
    on="UdtalelseId",
)

# Vi forbereder dataene til brug i modellen
topic_summaries = topic_summaries.groupby(["Sæson", "EmneNr"], as_index=False).agg(
    {"Opsummering": lambda x: " ".join(x)}
)
topic_summaries = topic_summaries.drop_duplicates()
topic_summaries["EmneId"] = topic_summaries["Sæson"] + "_" + topic_summaries["EmneNr"]

# Vi opsummerer kun data, som er nye
if prev_topic_auto_names is not None and not prev_topic_auto_names.empty:
    new_topics = topic_summaries[
        ~topic_summaries["EmneId"].isin(prev_topic_auto_names["Id"])
    ].copy()
else:
    new_topics = topic_summaries.copy()
n_new_topics = len(new_topics)

# Vi giver modellen følgende instruktioner
system_prompt = get_prompt("emnefortolkning.txt")

if n_new_topics:
    # Vi danner menneskevenlige emner pba. opsummeringerne for hvert emne
    full_texts = new_topics["Opsummering"].tolist()
    full_ids = new_topics["EmneId"].tolist()
    topic_auto_names = query_llm_multiple(
        openai_client, system_prompt, full_texts, full_ids, max_rpm=max_rpm
    )

    # Vi beholder kun relevante kolonner
    cols_to_rename = {"Response": "AutomatiskEmne"}
    cols_to_keep = ["Id", "AutomatiskEmne"]
    topic_auto_names = topic_auto_names.rename(columns=cols_to_rename)
    topic_auto_names = topic_auto_names[cols_to_keep]

    # Vi sammensætter tidligere og nuværende resultater i et datasæt
    topic_auto_names = pd.concat([prev_topic_auto_names, topic_auto_names])

    # Vi sorterer og eksporterer data
    topic_auto_names = topic_auto_names.sort_values("Id")
    topic_auto_names = topic_auto_names.reset_index(drop=True)
    topic_auto_names.to_parquet("output/results_llm_lda_topics.parquet")

    print("LDA emner er nu omdannet til menneskesprog.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Opsummering af lovforslagsæsoner [WIP as of 26-11-2024]


# %% Opsummering af partiernes holdninger [WIP as of 26-11-2024]

"""
OBS: Partiholdninger opsummeres på nyt hver gang brugeren kører
dette skript, hvis 'summarize_party_opinions = True'. Dette kan
medføre større omkostninger, og skal kun bruges hvis input dataene
ændrer sig.

Vi sammensætter alle udtalelser (opsummeringer) fra hvert enkelt
parti i en tekst, og beder OpenAI modellen om at opsummere partiets
politik på indfødsretsområdet. Vi skælner imellem nuværende
holdninger (baseret på de sidste 2 lovforslag, behandlet i FT), og
tidligere (historiske) holdninger.
"""

if summarize_party_opinions:
    print("Opsummering af partiernes holdninger er nu i gang...")

    # Nedenstående skal ikke betragtes som partigrupper i denne analyse
    not_parties = [
        "Uden for folketingsgrupperne (UFG)",
        "Udlændingeministeren",
        "Folketingets formand",
    ]

    # # Unikke partier, hvis holdninger skal opsummeres
    # all_parties = all_debates["PartiGruppe"].sort_values().unique().tolist()
    # all_parties = [p for p in all_parties if p not in not_parties]

    # Vi sætter sammen udtalelsernes opsummeringer med partigrupper
    party_opinions = speech_summaries.copy()
    party_opinions = pd.merge(
        party_opinions,
        all_debates[["UdtalelseId", "Sæson", "PartiGruppe"]],
        how="left",
        on="UdtalelseId",
    )
    party_opinions = party_opinions[
        ~party_opinions["PartiGruppe"].isin(not_parties)
    ].copy()

    # Vi betragter de sidste 2 sæsoner som "nuværende politik"
    all_debates["Dato"] = all_debates["Tid"].dt.date
    all_periods = all_debates[["Dato", "Sæson"]].sort_values(by="Dato")
    all_periods = all_periods.drop_duplicates("Sæson")
    all_periods = all_periods["Sæson"].tolist()
    current_periods = all_periods[-2:]
    past_periods = all_periods[:-2]
    party_opinions["HoldningType"] = np.where(
        party_opinions["Sæson"].isin(current_periods),
        "Nuværende holdning",
        "Historisk holdning",
    )
    party_opinions["Sæsoner"] = np.where(
        party_opinions["Sæson"].isin(current_periods),
        ", ".join(current_periods),
        "Alle andre",
    )

    # Vi forbereder dataene til brug i modellen
    party_opinions = party_opinions.groupby(
        ["PartiGruppe", "HoldningType", "Sæsoner"], as_index=False
    ).agg({"Opsummering": lambda x: " ".join(x)})
    party_opinions = party_opinions.drop_duplicates()
    party_opinions["Id"] = (
        party_opinions["PartiGruppe"]
        + "&&&"
        + party_opinions["HoldningType"]
        + "&&&"
        + party_opinions["Sæsoner"]
    )
    party_opinions["Opsummering"] = (
        "[Partinavn: "
        + party_opinions["PartiGruppe"]
        + "]\n"
        + party_opinions["Opsummering"]
    )

    # Vi skiler historiske og nuværende holdninger af
    tmp_current = party_opinions[party_opinions["HoldningType"] == "Nuværende holdning"]
    tmp_past = party_opinions[party_opinions["HoldningType"] == "Historisk holdning"]

    # Vi giver modellen følgende instruktioner
    system_prompt_current = get_prompt("partiholdning_nuværende.txt")
    system_prompt_past = get_prompt("partiholdning_tidligere.txt")

    # Vi opsummerer partiernes nuværende holdninger
    full_texts = tmp_current["Opsummering"].tolist()
    full_ids = tmp_current["Id"].tolist()
    party_opinion_current = query_llm_multiple(
        openai_client, system_prompt_current, full_texts, full_ids, max_rpm=max_rpm
    )

    # Vi opsummerer partiernes tidligere/historiske holdninger
    full_texts = tmp_past["Opsummering"].tolist()
    full_ids = tmp_past["Id"].tolist()
    party_opinion_past = query_llm_multiple(
        openai_client, system_prompt_current, full_texts, full_ids, max_rpm=max_rpm
    )

    # Vi sammensætter dataene og renser dem lidt [WIP as of 26-11-2024]
    cols_to_rename = {"Response": "Holdning"}
    cols_to_keep = ["PartiGruppe", "HoldningType", "Sæsoner", "Holdning"]
    # party_opinions = pd.concat([party_opinion_current, party_opinion_past])
    party_opinions = party_opinion_current.copy()  # temp as of 26-11-2024
    party_opinions["PartiGruppe"] = party_opinions["Id"].str.split("&&&").str[0]
    party_opinions["HoldningType"] = party_opinions["Id"].str.split("&&&").str[1]
    party_opinions["Sæsoner"] = party_opinions["Id"].str.split("&&&").str[2]
    party_opinions = party_opinions.rename(columns=cols_to_rename)
    party_opinions = party_opinions[cols_to_keep]

    """
    ==================
    Kiril, 26-11-2024:
    ==================
    I'm getting the following error message, which prevents me from
    completing the work on the historical opinions. I need 24 h
    before continuing my work...
    """

    # Vi eksporterer dataene til videre brug
    party_opinions.to_parquet("output/results_party_opinions.parquet")

    print("Partiernes holdninger er nu opsummeret.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Opsummering af enkelte folketingsmedlemmers holdninger [WIP as of 26-11-2024]

"""
==================
Kiril, 14-11-2024:
==================
Metoden anvendt nedenunder er en proof-of-concept tilgang,
som muligvis kræver visse justeringer for at få den til at
virke.

Kiril, 26-11-2024: Need to create a sorted list of FT medlemmer
for at finde ud af, om der har været nogle partihoppere. De skal
have deres nuværende (sidste) parti i navnet. Lav en df med navne,
og gennemgå den manuelt inden du fortsætter med at kode i denne del.
"""


# Vi giver modellen følgende instruktioner
system_prompt_current = get_prompt("personholdning_nuværende.txt")
system_prompt_prev = get_prompt("personholdning_tidligere.txt")

# Partigruppe, vi bruger som test
tmp_person = "Mikkel Bjørn (DF)"
person_name = f" Folketingsmedlemmets navn er {tmp_person}."
all_debates["Navn"].unique()
tmp_person_opinion = speech_summaries.copy()
tmp_person_opinion = pd.merge(
    tmp_person_opinion,
    all_debates[["UdtalelseId", "År", "Navn"]],
    how="left",
    on="UdtalelseId",
)
tmp_person_opinion = tmp_person_opinion[tmp_person_opinion["Navn"] == tmp_person].copy()

# Nuværende politik/holdninger
tmp_person_opinion_current = tmp_person_opinion[
    tmp_person_opinion["År"] == tmp_person_opinion["År"].max()
].copy()
tmp_person_opinion_current = tmp_person_opinion_current["Opsummering"].tolist()
tmp_person_opinion_current = " ".join(tmp_person_opinion_current)

results = query_llm_multiple(
    openai_client,
    system_prompt_current + person_name,
    [tmp_person_opinion_current],
    [tmp_person],
    max_rpm=max_rpm,
)
results["Response"].iloc[0]


# Tidligere politik/holdninger
tmp_person_opinion_prev = tmp_person_opinion[
    tmp_person_opinion["År"] != tmp_person_opinion["År"].max()
].copy()
tmp_person_opinion_prev["OpsummeringMedÅr"] = (
    tmp_person_opinion_prev["År"].astype(str)
    + ": "
    + tmp_person_opinion_prev["Opsummering"]
)
tmp_person_opinion_prev = tmp_person_opinion_prev["OpsummeringMedÅr"].tolist()
tmp_person_opinion_prev = " ".join(tmp_person_opinion_prev)

results = query_llm_multiple(
    openai_client,
    system_prompt_prev + person_name,
    [tmp_person_opinion_prev],
    [tmp_person],
    max_rpm=max_rpm,
)
results["Response"].iloc[0]


"""
==================
Kiril, 14-11-2024:
==================
Could be useful to get the top 20-40% based on the length of their
speeches (measured in N of characters) and then only apply it to
them. Alternatively, apply on all from the most recent years.
"""

# %% Endelig bekræftelse

print("Resultater fra tekstanalyse findes i 'output data' mappen.")
print("FÆRDIG.")

# %%
