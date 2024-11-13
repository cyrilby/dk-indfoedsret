"""
====================================
Tekstanalyse ved brug af OpenAIs LLM
====================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 13-11-2024

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

# # Vi markerer rækker med meget korte udtalelser
# all_debates["LangUdtalelse"] = all_debates["Udtalelse"].str.len() > max_length

# Vi markerer de resterende rækker, som skal bruges ifm. OpenAI modellen
all_debates["OpsummerUdtalelse"] = ~all_debates["ApolitiskUdtalelse"]


# %% Opsummering af alle relevante udtalelser [WIP as of 13-11-2024]

"""
Vi opsummerer alle politiske udtalelser, som er længere end en
sætning. Dvs. at Folketingets formands udtalelser samt meget korte
udtalelser vil ikke blive forkortet da de allerede er ret korte.
"""

print("Opsummering af alle relevante udtalelser er nu i gang...")

# Vi giver modellen følgende instruktioner
system_prompt = get_prompt("opsummering.txt")

# # Vi opsummerer kun data, som er nye
# if prev_speech_summaries is not None and not prev_speech_summaries.empty:
#     new_speeches = all_debates[
#         ~all_debates["UdtalelseId"].isin(prev_speech_summaries["UdtalelseId"])
#     ].copy()
# else:
#     new_speeches = all_debates.copy()
# n_new_speeches = len(new_speeches)

# WIP as of 13-11-2024: Testing summarization 1 year at a time
# Doing things 1 year at a time
new_speeches = all_debates[all_debates["År"] == 2006].copy()
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
        "violence": "SprobgrugVold",
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

    # Vi sorterer og eksporterer data
    speech_summaries = speech_summaries.sort_values("UdtalelseId")
    speech_summaries = speech_summaries.reset_index(drop=True)
    speech_summaries.to_parquet("output/results_llm_opsummering.parquet")

    print("Automatisk opsummering af udtalelser færdig.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% LDA emner til menneskesprog [WIP as of 13-11-2024]

"""
=============================
KBO kommentar fra 13-11-2024:
=============================
Udviklingen i denne del kræver, at vi først fikser det med
opsummeringen.

OBS: Det vil være godt også at eskludere apolitiske udtalelser
fra den almindelige tekstanalyse (hvis de ikke allerede er
eksluderet af den).

Der er følgende alternativer, som jeg kan teste:
1) Udvalg fx de top 25% af alle ord for hvert emne (baseret på
ordenes vægt), og spørg ChatGPT om at lave en overskrift.
2) Udvalg fx de top 3/5/10 mest precise udtalelser for hvert
emne (baseret på "Sandsynlighed" kolonnen), og spørg ChatGPT om
at lave en overskrift.
"""


# Fx - denne kode er pastet fra det andet tekstanalyse skript

# # For hvert emne finder vi de 5 udtalelser med højest præcision
# n_samples = 5
# sort_vars = ["Sæson", "EmneNr", "Sandsynlighed"]
# sort_vars_asc = [True, True, False]
# group_vars = ["Sæson", "EmneNr"]
# best_fitting = auto_topics.copy()
# best_fitting = best_fitting.sort_values(by=sort_vars, ascending=sort_vars_asc)
# best_fitting["ScoreRank"] = best_fitting.groupby(group_vars).cumcount() + 1
# best_fitting = best_fitting[best_fitting["ScoreRank"] <= n_samples].copy()
# best_fitting = pd.merge(
#     best_fitting,
#     all_debates[["UdtalelseId", "Udtalelse"]],
#     how="left",
#     on="UdtalelseId",
# )


# %% Endelig bekræftelse

print("Resultater fra tekstanalyse findes i 'output data' mappen.")
print("FÆRDIG.")

# %%
