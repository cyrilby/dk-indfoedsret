"""
==========================================
Tekstanalyse ved brug af klassiske metoder
==========================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 24-07-2025

Formålet ved dette skript er at tage de rensede data fra debatterne,
som indeholder diverse taler ("udtalelser"), og bruge dem til klassisk
tekstanalyse, specifikt automatisk gruppering af udtalelserne i
diverse kategorier. Alt i den her skript foregår lokalt uden at
bruge eksterne API'er.
"""

# %% Generel opsætning

# Import af relevante pakker
import os
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from tqdm import tqdm
from typing import Tuple

# Import af data, som allerede er blevet renset
all_debates = pd.read_parquet("output/ft_behandlinger.parquet")
speech_tokens = pd.read_parquet("output/results_speech_tokens.parquet")
mapping_roller = pd.read_excel("input/mapping_tabeller.xlsx", sheet_name="Partigrupper")

# Import af modeltest og emner fra tidligere kørsler af skriptet
file_path_model_tests = "output/results_topics_model_stats.parquet"
file_path_auto_topics = "output/results_topics_auto.parquet"
file_path_auto_words = "output/results_words_auto.parquet"
file_path_manual_topics = "output/results_topics_manual.parquet"
prev_model_tests = (
    pd.read_parquet(file_path_model_tests)
    if os.path.exists(file_path_model_tests)
    else None
)
prev_auto_topics = (
    pd.read_parquet(file_path_auto_topics)
    if os.path.exists(file_path_auto_topics)
    else None
)
prev_auto_words = (
    pd.read_parquet(file_path_auto_words)
    if os.path.exists(file_path_auto_words)
    else None
)
prev_manual_topics = (
    pd.read_parquet(file_path_manual_topics)
    if os.path.exists(file_path_manual_topics)
    else None
)

# Vi skal kun bruge politiske udtalelser, så vi sorterer
# folketingets formands taler fra
condition_1 = ~(all_debates["PartiGruppe"] == "Folketingets formand")
condition_2 = ~(all_debates["Rolle"].str.lower().str.contains("formand"))
relevant_speeches = all_debates[(condition_1) | (condition_2)]["UdtalelseId"].tolist()
speech_tokens = speech_tokens[
    speech_tokens["UdtalelseId"].isin(relevant_speeches)
].copy()

# Vi tilføjer en kolonne, som viser lovforslagssæson
speech_tokens["Sæson"] = (
    speech_tokens["UdtalelseId"].str.split("-").str[:2].str.join("-")
)


# %% Egen funktion til beregning af "coherence score" for LDA modeller


def compute_coherence_score(ldamodel, dictionary, model_tokens):
    coherence_model_lda = CoherenceModel(
        model=ldamodel, texts=model_tokens, dictionary=dictionary, coherence="c_v"
    )
    coherence_score = coherence_model_lda.get_coherence()
    return coherence_score


# %% Egen funktion til beregning af "perplexity score" for LDA modeller


def compute_perplexity_score(ldamodel, doc_term_matrix):
    """
    Compute the perplexity score for a given LDA model.

    Args:
        ldamodel (LdaModel): Trained LDA model.
        doc_term_matrix (list): Bag-of-words representation of the documents.

    Returns:
        float: Perplexity score of the LDA model.
    """
    perplexity_score = ldamodel.log_perplexity(doc_term_matrix)
    return perplexity_score


# %% Egen funktion til test af enkelt LDA model


def fit_lda_model(
    input_df: pd.DataFrame,
    tokens_col: str,
    id_col: str,
    n_topics: int,
    calc_score: bool = False,
    return_all: bool = False,
    print_all: bool = False,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Tager en nested liste med ord og grupperer listerne med ord på øverste
    niveau i diverse temaer (emner) pba. de ord, listene indeholder.
    Returnerer et datasæt med den mest sandsylige emner for enhver liste,
    samt den beregnede sandsynlighed.

    Args:
        input_df (pd.DataFrame): datasæt, hvoraf en af kolonnerne indeholder
        lister med enkelte ord (efter lemmatization og tokenization er udøvet)
        tokens_col (str): navn på kolonnen, som indeholder de enkelte tokens
        id_col (str): navn på kolonnen, som indeholder ID-oplysningerne
        n_topics (int): antal emner at bruge i modellen
        calc_score (bool, optional): om perplexity score burde
        beregnes når modellen er blevet testet. Pr. definition False
        return_all (bool, optional): om alle foreslåede temaer (emner)
        skal ses i outputtet. Pr. definition 'False'.
        print_all (bool, optional): om mere information om modellen og dens
        resultater skal udskrives undervejs. Per definition 'False'.

    Returns:
        Tuple[LdaModel, pd.DataFrame, pd.DataFrame]: coherence_score + datasæt med de
        endelige resultater fra modellen, herunder de korrekte emner for hver liste
        med enkelte ord, samt datasæt med de ord, som indgår i de diverse emner
    """

    # Vi må ikke have blanke felter
    topic_tokens = input_df.fillna("").copy()

    # Vi konverterer tokens til en nested liste for hver udtalelse
    model_tokens = topic_tokens.groupby(id_col)[tokens_col].apply(list).tolist()
    id_tokens = topic_tokens[id_col].unique().tolist()

    # Vi forbereder dataene til brug i en LDA model
    dictionary = corpora.Dictionary(model_tokens)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in model_tokens]

    # Vi træner en LDA model med N antal emner og en fikseret "random_state",
    # som sikrer, at resultaterne kan genskabes
    Lda = LdaModel
    if print_all:
        print(f"Test af LDA tekstmodel med {n_topics} emner er nu i gang...")
    ldamodel = Lda(
        doc_term_matrix,
        num_topics=n_topics,
        id2word=dictionary,
        passes=50,
        random_state=173,
    )

    # # Vi beregner "coherence score" for modellen
    # coherence_score = compute_coherence_score(
    #     ldamodel, doc_term_matrix, dictionary, model_tokens
    # )

    # Vi beregner "perplexity score" for modellen
    if calc_score:
        perplexity_score = compute_perplexity_score(ldamodel, doc_term_matrix)
    else:
        perplexity_score = None

    # Her finder vi de automatisk skabte emner
    topics = ldamodel.show_topics(num_topics=n_topics, num_words=None, formatted=False)
    topics_words = []
    for topic_id, topic_words in topics:
        for word, weight in topic_words:
            topics_words.append((topic_id, word, weight))
    topics_words = pd.DataFrame(topics_words, columns=["EmneId", "Ord", "Vægt"])

    # Her skaber vi temaer i de oprindelige data
    auto_topics = [ldamodel.get_document_topics(bow) for bow in doc_term_matrix]

    # Her konverterer vi resultaterne til et datasæt
    topics_df_data = []
    for doc_topics in auto_topics:
        topic_dict = {f"Topic {topic_id}": prob for topic_id, prob in doc_topics}
        topics_df_data.append(topic_dict)
    auto_topics = pd.DataFrame(topics_df_data)
    auto_topics = auto_topics.fillna(0)

    # Her finder vi ud af, hvilket tema har den højeste sandsynlighed
    auto_topics = auto_topics.apply(pd.to_numeric, errors="coerce")
    auto_topics["EmneNr"] = auto_topics.idxmax(axis=1)
    auto_topics["Sandsynlighed"] = auto_topics.select_dtypes(include="number").max(
        axis=1
    )

    # Vi tilføjer informationer såsom ID til outputtet
    auto_topics[id_col] = id_tokens

    # Vi beholder kun de mest relevante emner medmindre andet er specificeret
    if not return_all:
        auto_topics = auto_topics[[id_col, "EmneNr", "Sandsynlighed"]]

    # Vi beregner mediansandsynligheden på tværs af dataene og giver data retur
    if print_all:
        med_prob = round(100 * auto_topics["Sandsynlighed"].median(), 1)
        print(f"Test af LDA tekstmodel med {n_topics} emner færdig.")
        print(f"Mediansandsynlighed for korrekt klassificering: {med_prob}%.")
    return perplexity_score, auto_topics, topics_words


# %% Automatisk test af LDA emnemodeller

print("Automatisk test af LDA emnemodeller er nu i gang...")

"""
Obs: LDA modelleringen sker på tværs af hele datasættet for
at kunne identificere emner, som gentages over tid. Pga. denne
tilgang er det nødvendigt med LLM-assisteret mapping af emnerne
hver gang, der kommer nye data fra Folketinget (2 gange om året).
"""

# Brug dette til at tvinge LDA analysen selv om der ikke er nye data
force_lda = False

# Vi noterer den sidste sæson, som er dækket af dataenee
latest_season = speech_tokens["Sæson"].iloc[-1]

# Vi tjekker om der er kommet nye data fra Folketinget: hvis ikke,
# så springer vi over LDA emneanalysen
latest_lda_season = prev_auto_topics["SæsonOpdateret"].iloc[0]
new_data_available = latest_lda_season != latest_season

if new_data_available or force_lda:
    # Vi tester modeller med mellem 2-15 emner i alt
    n_topics = np.arange(2, 16)
    median_probability = []
    perplexity_score = []
    topics_for_df = []

    # Resultaterne af de enkelte tests bliver noteret i et særskilt df
    for tmp_topic in tqdm(n_topics, total=len(n_topics)):
        perpl_score, auto_topics, _ = fit_lda_model(
            speech_tokens, "Token", "UdtalelseId", tmp_topic, True
        )
        med_prob = auto_topics["Sandsynlighed"].median()
        median_probability.append(med_prob)
        perplexity_score.append(perpl_score)
        topics_for_df.append(tmp_topic)

    # Vi opsummerer resultaterne af vores tests
    model_tests = pd.DataFrame(
        {
            "AntalEmner": topics_for_df,
            "MedianResultat": median_probability,
            "PerplexityScore": perplexity_score,
        }
    )
    model_tests.insert(0, "SæsonOpdateret", latest_season)

    # Vi sorterer data
    model_tests = model_tests.sort_values(["SæsonOpdateret", "PerplexityScore"])
    model_tests = model_tests.reset_index(drop=True)

    print("Automatisk test af LDA emnemodeller færdig.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Gruppering af udtalelser i diverse emner

print("Gruppering af udtalelser i diverse emner er nu i gang...")

if new_data_available or force_lda:
    # Vi vælger den bedste model og bruger den til at gruppere udtalelserne i emner
    best_model_n = model_tests["AntalEmner"].iloc[0]
    _, auto_topics, auto_words = fit_lda_model(
        speech_tokens, "Token", "UdtalelseId", best_model_n
    )
    auto_topics.insert(0, "SæsonOpdateret", latest_season)
    auto_words.insert(0, "SæsonOpdateret", latest_season)

    # Vi sorterer data
    auto_topics = auto_topics.sort_values("UdtalelseId")
    auto_topics = auto_topics.reset_index(drop=True)
    auto_words = auto_words.sort_values(
        by=["SæsonOpdateret", "EmneId", "Vægt"], ascending=[True, True, False]
    )
    auto_words = auto_words.reset_index(drop=True)

    print("Gruppering af udtalelser i diverse emner færdig.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Manuelle tjek for, at visse ord bliver nævnt

"""
Her undersøger vi blot om nogle specifikke ord bliver nævnt ifm. hver
eneste udtalelse. Det er fx referencer til kriminalitet, udlændingenes
bidrag til/integration i det danske samfund osv.
"""

print("Udfører manuelle tjek for, at visse ord bliver nævnt...")

# Emner, vi gerne vil tjekke dataene for
topics_to_check = {
    "Kriminalitet": "kriminal|kriminel|forbryd",
    "Integration": "integration|integrer|parallelsamfund",
    "Religion": "kristendom|judaisme|islam|muslim|kristen|jøde",
    "Værdier": "værdi|respekt|grundlov",
    "Bidrag": "bidrag|bidrog|arbejde|job",
    "Fortjeneste": "gave|beløn|fortjen",
    "Sværhed": "krav|stram|streng|svært|nemt|betinge",
    "Værn": "værne|beskytte|bevare",
}

# Unikke lovforslgassæsoner, som kræver optælling af visse ord
new_seasons = speech_tokens["Sæson"].unique().tolist()

# Vi grupperer emner kun for nye lovforslagssæsoner
if prev_manual_topics is not None and not prev_manual_topics.empty:
    tested_seasons = prev_manual_topics["Sæson"].unique().tolist()
else:
    tested_seasons = []
new_seasons = [season for season in new_seasons if season not in tested_seasons]
n_seasons_count = len(new_seasons)

if n_seasons_count:
    # Tjek for, om ovennævnte ord nævnes i de individuelle udtalelser
    manual_topics = speech_tokens.copy()
    for var_name, var_val in zip(topics_to_check.keys(), topics_to_check.values()):
        manual_topics[var_name] = manual_topics["Token"].str.contains(var_val)

    # Vi konverterer datane fra "bredt" til "langt" format
    manual_topics = pd.melt(
        manual_topics,
        id_vars=["Sæson", "UdtalelseId"],
        value_vars=manual_topics.columns[2:],
        value_name="ReferererTilEmne",
        var_name="Emne",
    )
    manual_topics["ReferererTilEmne"] = manual_topics["ReferererTilEmne"].astype(int)

    print("Manuelle tjek for, at visse ord bliver nævnt, er færdig.")

else:
    print("Obs: Springer over pga. mangel af nye input data.")


# %% Eksport af de færdige analyser

# Vi eksporterer datasæt med outputs kun hvis der er nye data
if new_data_available or force_lda:
    model_tests.to_parquet("output/results_topics_model_stats.parquet")
    auto_topics.to_parquet("output/results_topics_auto.parquet")
    auto_words.to_parquet("output/results_words_auto.parquet")
if n_seasons_count:
    manual_topics.to_parquet("output/results_topics_manual.parquet")

print("Resultater fra tekstanalyse findes i 'output data' mappen.")
print("FÆRDIG.")

# %%
