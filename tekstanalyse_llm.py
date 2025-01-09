"""
====================================
Tekstanalyse ved brug af OpenAIs LLM
====================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 09-01-2025

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

# Fjern advarsler, når .ffill() eller .bfill() anvendes
pd.set_option("future.no_silent_downcasting", True)

# Om visse dele af analysen skal opsummeres
# (Gælder kun i tilfælde, hvor ny data ikke trackes automatisk)
# OBS: Sat til "True" hvis vi har nye input data
summarize_party_opinions = False
summarize_personal_opinions = True

# Maks antal requests pr. minut for Azure OpenAI API
max_rpm = None  # or 20

# Maks længde for tekster, som ikke skal opsummmeres (antal tegn)
max_length = 160

# Import af renset input data
all_debates = pd.read_parquet("output/ft_behandlinger.parquet")
speech_tokens = pd.read_parquet("output/results_speech_tokens.parquet")
latest_parties = pd.read_parquet("output/ft_seneste_partigruppe.parquet")

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
    topic_auto_names.to_excel(
        "output/results_llm_lda_topics.xlsx", index=False
    )  # til brug for mapping

    print("LDA emner er nu omdannet til menneskesprog.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Opsummering af partiernes holdninger

"""
OBS: Partiholdninger opsummeres på nyt hver gang brugeren kører
dette skript, hvis 'summarize_party_opinions = True'. Dette kan
medføre større omkostninger, og skal kun bruges hvis input dataene
ændrer sig.

-> Tilgang til nuværende holdning: alle udtalelser fra partiet
opsummeres i en kort tekst.
-> Tilgang til historiske holdninger: for hvert parti og sæson, vi
vælger de top 3 udtalelser (pba. deres længde i antal ord), og
vi opsummerer kun disse.
"""

if summarize_party_opinions:
    print("Opsummering af partiernes holdninger er nu i gang...")

    # Nedenstående skal ikke betragtes som partigrupper i denne analyse
    not_parties = [
        "Uden for folketingsgrupperne (UFG)",
        "Udlændingeministeren",
        "Folketingets formand",
    ]

    # Vi sætter sammen udtalelsernes opsummeringer med partigrupper
    party_opinions = speech_summaries.copy()
    party_opinions = pd.merge(
        party_opinions,
        all_debates[["UdtalelseId", "Sæson", "PartiGruppe", "UdtalelseLængdeOrd"]],
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

    # Når det kommer til historiske data, vi udvælger de 3 længste
    # taler for hvert parti og sæson, og vi kigger kun på de sidste 10 år.
    # På denne måde får vi de mest relevante data, mens vi reducerer
    # forbruget af OpenAI tokens og undgår API fejl og begrænsninger.
    party_opinions = party_opinions.sort_values(
        by=["PartiGruppe", "Sæson", "UdtalelseLængdeOrd"], ascending=[True, True, False]
    )
    party_opinions = party_opinions.reset_index(drop=True)
    party_opinions["UdtalelseNr"] = (
        party_opinions.groupby(["PartiGruppe", "Sæson"]).cumcount() + 1
    )
    party_opinions["År"] = party_opinions["Sæson"].str.slice(0, 4).astype(int)
    latest_year = np.max(party_opinions["År"])
    relevant_years = np.arange(latest_year - 10, latest_year + 1)
    party_opinions["RækkenSkalOpsummeres"] = np.where(
        (party_opinions["HoldningType"] == "Historisk holdning")
        & (party_opinions["UdtalelseNr"] > 3),
        False,
        True,
    )
    party_opinions["RækkenSkalOpsummeres"] = np.where(
        party_opinions["År"].isin(relevant_years),
        party_opinions["RækkenSkalOpsummeres"],
        False,
    )
    party_opinions = party_opinions[party_opinions["RækkenSkalOpsummeres"]].copy()

    # For historiske holdninger, vi tilføjer årene til teksten
    party_opinions["Opsummering"] = np.where(
        party_opinions["HoldningType"] == "Historisk holdning",
        "[År: "
        + party_opinions["År"].astype(str)
        + " ]: "
        + party_opinions["Opsummering"],
        party_opinions["Opsummering"],
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
    system_prompt_prev = get_prompt("partiholdning_tidligere.txt")

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
        openai_client, system_prompt_prev, full_texts, full_ids, max_rpm=max_rpm
    )

    # Vi sammensætter dataene og renser dem lidt
    cols_to_rename = {"Response": "Holdning"}
    cols_to_keep = ["PartiGruppe", "HoldningType", "Sæsoner", "Holdning"]
    party_opinions = pd.concat([party_opinion_current, party_opinion_past])
    party_opinions = party_opinions.reset_index(drop=True)
    party_opinions["PartiGruppe"] = party_opinions["Id"].str.split("&&&").str[0]
    party_opinions["HoldningType"] = party_opinions["Id"].str.split("&&&").str[1]
    party_opinions["Sæsoner"] = party_opinions["Id"].str.split("&&&").str[2]
    party_opinions = party_opinions.rename(columns=cols_to_rename)
    party_opinions = party_opinions[cols_to_keep]

    # Vi eksporterer dataene til videre brug
    party_opinions.to_parquet("output/results_party_opinions.parquet")

    print("Partiernes holdninger er nu opsummeret.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Opsummering af enkelte folketingsmedlemmers holdninger

"""
Metoden anvendt nedenunder er den samme som i den forrige
sektion, med den forskel, at vi kigger på individuelle
holdninger her, og at vi også tager potentielle partihoppere
i betragtning for at undgå dobbel optælling af samme person.
"""


if summarize_personal_opinions:
    print("Opsummering af folketingsmedlemmernes holdninger er nu i gang...")
    # Først tilpasser vi dataene vedr. seneste partigrupper
    cols_to_keep = ["Sæson", "Navn", "NuværendePartiGruppe", "SenestePartiGruppe"]
    new_col_names = {"PartiGruppe": "NuværendePartiGruppe"}
    latest_parties["Navn"] = (
        latest_parties["Navn"] + " (" + latest_parties["PartiGruppeKort"] + ")"
    )
    latest_parties["Sæson"] = (
        latest_parties["År"].astype(str) + "-" + latest_parties["Sæson"].str.slice(0, 1)
    )

    latest_parties = latest_parties.rename(columns=new_col_names)
    latest_parties = latest_parties[cols_to_keep]

    # Derefter kobler vi dem til dataene vedr. behandlingerne
    cols_for_merge = ["Sæson", "Navn"]
    all_debates = pd.merge(all_debates, latest_parties, how="left", on=cols_for_merge)

    # Vi sikrer, at personnavnet ikke indeholder partiet
    all_debates["PersonNavn"] = all_debates["Navn"].str.extract(r"^(.*?)\s\(")
    all_debates["PersonNavn"] = all_debates["PersonNavn"].fillna(all_debates["Navn"])

    # Vi sikrer, at ingen politiske udtalelser mangler seneste parti
    all_debates["SenestePartiGruppe"] = all_debates.groupby("PersonNavn")[
        "SenestePartiGruppe"
    ].transform(lambda x: x.ffill().bfill())

    # Vi giver modellen følgende instruktioner
    system_prompt_current = get_prompt("personholdning_nuværende.txt")
    system_prompt_prev = get_prompt("personholdning_tidligere.txt")

    # Vi looper over alle unikke taler udover Folketingets formands
    # og udlændingeministerens udtalelser
    political_speech = ~all_debates["ApolitiskUdtalelse"]
    personal_opinions = all_debates[political_speech].copy()
    unique_speakers = personal_opinions["PersonNavn"].unique()
    n_unique_speakers = len(unique_speakers)

    # Vi betragter de sidste 2 sæsoner som "nuværende politik"
    all_debates["Dato"] = all_debates["Tid"].dt.date
    all_periods = all_debates[["Dato", "Sæson"]].sort_values(by="Dato")
    all_periods = all_periods.drop_duplicates("Sæson")
    all_periods = all_periods["Sæson"].tolist()
    current_periods = all_periods[-2:]
    past_periods = all_periods[:-2]
    personal_opinions["HoldningType"] = np.where(
        personal_opinions["Sæson"].isin(current_periods),
        "Nuværende holdning",
        "Historisk holdning",
    )
    personal_opinions["Sæsoner"] = np.where(
        personal_opinions["Sæson"].isin(current_periods),
        ", ".join(current_periods),
        "Alle andre",
    )

    # Vi tilføjer udtalelsernes opsummeringer til dataene
    personal_opinions = pd.merge(
        personal_opinions, speech_summaries, how="left", on="UdtalelseId"
    )

    # Når det kommer til historiske data, vi udvælger de 3 længste
    # taler for hver person og sæson, og vi kigger kun på de sidste 10 år.
    # På denne måde får vi de mest relevante data, mens vi reducerer
    # forbruget af OpenAI tokens og undgår API fejl og begrænsninger.
    personal_opinions = personal_opinions.sort_values(
        by=["PersonNavn", "Sæson", "UdtalelseLængdeOrd"], ascending=[True, True, False]
    )
    personal_opinions = personal_opinions.reset_index(drop=True)
    personal_opinions["UdtalelseNr"] = (
        personal_opinions.groupby(["PersonNavn", "Sæson"]).cumcount() + 1
    )
    personal_opinions["År"] = personal_opinions["Sæson"].str.slice(0, 4).astype(int)
    latest_year = np.max(personal_opinions["År"])
    relevant_years = np.arange(latest_year - 10, latest_year + 1)
    personal_opinions["RækkenSkalOpsummeres"] = np.where(
        (personal_opinions["HoldningType"] == "Historisk holdning")
        & (personal_opinions["UdtalelseNr"] > 3),
        False,
        True,
    )
    personal_opinions["RækkenSkalOpsummeres"] = np.where(
        personal_opinions["År"].isin(relevant_years),
        personal_opinions["RækkenSkalOpsummeres"],
        False,
    )
    personal_opinions = personal_opinions[
        personal_opinions["RækkenSkalOpsummeres"]
    ].copy()

    # For historiske holdninger, vi tilføjer årene til teksten
    personal_opinions["Opsummering"] = np.where(
        personal_opinions["HoldningType"] == "Historisk holdning",
        "[År: "
        + personal_opinions["År"].astype(str)
        + " ]: "
        + personal_opinions["Opsummering"],
        personal_opinions["Opsummering"],
    )

    # For hver person, vi noterer alle relevante partigrupper
    parties_for_person = personal_opinions.groupby("PersonNavn")["PartiGruppe"].apply(
        lambda x: ", ".join(sorted(x.unique()))
    )
    personal_opinions = personal_opinions.merge(
        parties_for_person.rename("AllePartiGrupper"), on="PersonNavn"
    )

    # Vi laver en kolonne med navn og alle partigrupper
    personal_opinions["NavnOgPartier"] = (
        "Folketingsmedlemmet hedder "
        + personal_opinions["PersonNavn"]
        + ". Han/hun har senest været medlem af "
        + personal_opinions["SenestePartiGruppe"]
        + ". Han/hun har tidligere været (medlem af) "
        + personal_opinions["AllePartiGrupper"]
    )

    # Vi forbereder dataene til brug i modellen
    personal_opinions = personal_opinions.groupby(
        ["PersonNavn", "NavnOgPartier", "HoldningType", "Sæsoner"], as_index=False
    ).agg({"Opsummering": lambda x: " ".join(x)})
    personal_opinions = personal_opinions.drop_duplicates()
    personal_opinions["Id"] = (
        personal_opinions["PersonNavn"]
        + "&&&"
        + personal_opinions["HoldningType"]
        + "&&&"
        + personal_opinions["Sæsoner"]
    )
    personal_opinions["Opsummering"] = (
        "["
        + personal_opinions["NavnOgPartier"]
        + "]\n"
        + personal_opinions["Opsummering"]
    )

    # Vi skiler historiske og nuværende holdninger af
    tmp_current = personal_opinions[
        personal_opinions["HoldningType"] == "Nuværende holdning"
    ]
    tmp_past = personal_opinions[
        personal_opinions["HoldningType"] == "Historisk holdning"
    ]

    # Vi opsummerer folketingsmedlemmernes nuværende holdninger
    full_texts = tmp_current["Opsummering"].tolist()
    full_ids = tmp_current["Id"].tolist()
    personal_opinion_current = query_llm_multiple(
        openai_client, system_prompt_current, full_texts, full_ids, max_rpm=max_rpm
    )

    # Vi opsummerer folketingsmedlemmernes historiske holdninger
    full_texts = tmp_past["Opsummering"].tolist()
    full_ids = tmp_past["Id"].tolist()
    personal_opinion_past = query_llm_multiple(
        openai_client, system_prompt_prev, full_texts, full_ids, max_rpm=max_rpm
    )

    # Vi sammensætter dataene og renser dem lidt
    cols_to_rename = {"Response": "Holdning"}
    cols_to_keep = ["PersonNavn", "HoldningType", "Sæsoner", "Holdning"]
    personal_opinions = pd.concat([personal_opinion_current, personal_opinion_past])
    personal_opinions = personal_opinions.reset_index(drop=True)
    personal_opinions["PersonNavn"] = personal_opinions["Id"].str.split("&&&").str[0]
    personal_opinions["HoldningType"] = personal_opinions["Id"].str.split("&&&").str[1]
    personal_opinions["Sæsoner"] = personal_opinions["Id"].str.split("&&&").str[2]
    personal_opinions = personal_opinions.rename(columns=cols_to_rename)
    personal_opinions = personal_opinions[cols_to_keep]

    # Vi eksporterer dataene til videre brug
    personal_opinions.to_parquet("output/results_personal_opinions.parquet")
    print("Folketingsmedlemmernes holdninger er nu opsummeret.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Endelig bekræftelse

print("Resultaterne fra tekstanalyse findes i 'output data' mappen.")
print("FÆRDIG.")

# %%
