"""
====================================
Tekstanalyse ved brug af OpenAIs LLM
====================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 14-11-2024

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


# %% LDA emner til menneskesprog [WIP as of 14-11-2024]

"""
==================
Kiril, 14-11-2024:
==================
Metoden anvendt nedenunder er en proof-of-concept tilgang,
som muligvis kræver visse justeringer for at få den til at
virke.
"""

# Fx - denne kode er pastet fra det andet tekstanalyse skript

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
best_fitting = pd.merge(
    best_fitting,
    speech_summaries[["UdtalelseId", "Opsummering"]],
    how="left",
    on="UdtalelseId",
)

# WIP: Udvid med sikkerhedsfiltre for at undgå triggering af OpenAIs sikkerheds mekanisme

# Vi giver modellen følgende instruktioner
system_prompt = get_prompt("emnefortolkning.txt")

# Vi tester tilgangen med et emne
tmp_topic = best_fitting[
    (best_fitting["Sæson"] == "2004-E") & (best_fitting["EmneNr"] == "Topic 2")
].copy()
tmp_topic = tmp_topic["Opsummering"].tolist()
tmp_topic = " ".join(tmp_topic)
print(tmp_topic)

results = query_llm_multiple(
    openai_client, system_prompt, [tmp_topic], ["Topic 0"], max_rpm=max_rpm
)
results["Response"].iloc[0]


# %% Opsummering af lovforslagsæsoner [WIP as of 14-11-2024]


# %% Opsummering af partienes holdninger [WIP as of 14-11-2024]

"""
==================
Kiril, 14-11-2024:
==================
Metoden anvendt nedenunder er en proof-of-concept tilgang,
som muligvis kræver visse justeringer for at få den til at
virke.
"""

# Vi giver modellen følgende instruktioner
system_prompt_current = get_prompt("partiholdning_nuværende.txt")
system_prompt_prev = get_prompt("partiholdning_tidligere.txt")

# Partigruppe, vi bruger som test
tmp_group = "Alternativet (ALT)"
all_debates["PartiGruppe"].unique()
tmp_group_opinion = speech_summaries.copy()
tmp_group_opinion = pd.merge(
    tmp_group_opinion,
    all_debates[["UdtalelseId", "År", "PartiGruppe"]],
    how="left",
    on="UdtalelseId",
)
tmp_group_opinion = tmp_group_opinion[
    tmp_group_opinion["PartiGruppe"] == tmp_group
].copy()

# Nuværende politik/holdninger
tmp_group_opinion_current = tmp_group_opinion[
    tmp_group_opinion["År"] == tmp_group_opinion["År"].max()
].copy()
tmp_group_opinion_current = tmp_group_opinion_current["Opsummering"].tolist()
tmp_group_opinion_current = " ".join(tmp_group_opinion_current)

results = query_llm_multiple(
    openai_client,
    system_prompt_current,
    [tmp_group_opinion_current],
    [tmp_group],
    max_rpm=max_rpm,
)
results["Response"].iloc[0]


# Tidligere politik/holdninger
tmp_group_opinion_prev = tmp_group_opinion[
    tmp_group_opinion["År"] != tmp_group_opinion["År"].max()
].copy()
tmp_group_opinion_prev["OpsummeringMedÅr"] = (
    tmp_group_opinion_prev["År"].astype(str)
    + ": "
    + tmp_group_opinion_prev["Opsummering"]
)
tmp_group_opinion_prev = tmp_group_opinion_prev["OpsummeringMedÅr"].tolist()
tmp_group_opinion_prev = " ".join(tmp_group_opinion_prev)

results = query_llm_multiple(
    openai_client,
    system_prompt_prev,
    [tmp_group_opinion_prev],
    [tmp_group],
    max_rpm=max_rpm,
)
results["Response"].iloc[0]


# %% Opsummering af enkelte folketingsmedlemmers holdninger [WIP as of 14-11-2024]

"""
==================
Kiril, 14-11-2024:
==================
Metoden anvendt nedenunder er en proof-of-concept tilgang,
som muligvis kræver visse justeringer for at få den til at
virke.
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
