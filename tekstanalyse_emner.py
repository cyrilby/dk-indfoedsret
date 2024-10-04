"""
==============================================
Tekstanalyse af taler under debatterne (emner)
==============================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 03-10-2024

Formålet ved dette skript er at tage de rensede data fra debatterne,
som indeholder diverse taler ("udtalelser"), og bruge dem til klassisk
tekstanalyse, specifikt automatisk gruppering af udtalelserne i diverse
kategorier. Alt i den her skript foregår lokalt uden at bruge eksterne API'er.
"""

# %% Generel opsætning

# Import af relevante pakker
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
mapping_topics = pd.read_excel("input/mapping_tabeller.xlsx", sheet_name="Emner")[
    ["EmneNr", "Emne"]
]

# Vi skal kun bruge politiske udtalelser, så vi sorterer
# folketingets formands taler fra
relevant_speeches = all_debates[all_debates["PartiGruppe"] != "Folketingets formand"][
    "UdtalelseId"
].tolist()
speech_tokens = speech_tokens[
    speech_tokens["UdtalelseId"].isin(relevant_speeches)
].copy()


# %% Egen funktion til beregning af "coherence score" for LDA modeller


def compute_coherence_score(ldamodel, doc_term_matrix, dictionary, model_tokens):
    coherence_model_lda = CoherenceModel(
        model=ldamodel, texts=model_tokens, dictionary=dictionary, coherence="c_v"
    )
    coherence_score = coherence_model_lda.get_coherence()
    return coherence_score


# %% Egen funktion til test af enkelt LDA model


def fit_lda_model(
    input_df: pd.DataFrame,
    tokens_col: str,
    id_col: str,
    n_topics: int,
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

    # Vi beregner "coherence score" for modellen
    coherence_score = compute_coherence_score(
        ldamodel, doc_term_matrix, dictionary, model_tokens
    )

    # Her finder vi de automatisk skabte emner
    topics = ldamodel.show_topics(num_topics=n_topics, formatted=False)
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
    return coherence_score, auto_topics, topics_words


# %% Automatisk test af LDA emnemodeller

print("Automatisk test af LDA emnemodeller er nu i gang...")

# Vi tester modeller med mellem 5 og 20 emner for at finde den mest præcise
n_topics = np.arange(5, 21)
median_probability = []
coherence_score = []

for tmp_n in tqdm(n_topics, total=len(n_topics)):
    coh_score, auto_topics, _ = fit_lda_model(
        speech_tokens, "Token", "UdtalelseId", tmp_n
    )
    med_prob = auto_topics["Sandsynlighed"].median()
    median_probability.append(med_prob)
    coherence_score.append(coh_score)

# Vi opsummerer resultaterne af vores tests
model_tests = pd.DataFrame(
    {
        "AntalEmner": n_topics,
        "MedianResultat": median_probability,
        "CoherenceScore": coherence_score,
    }
)
model_tests = model_tests.sort_values("CoherenceScore", ascending=False)
model_tests = model_tests.reset_index(drop=True)

print("Automatisk test af LDA emnemodeller færdig. Her er resultaterne:")
print(model_tests.head(len(model_tests)))


# %% Gruppering af udtalelser i diverse temaer

"""
=================
Kiril's comments:
=================
Use a package that supports Danish-language tests to assign all
speeches into somewhere between 10-20 themes. Vary the number of
themes and their parameters until you get the highest mean
probability that each speech has been assigned to the correct
category. Then, get the 3-4 speeches within each team that have
the highest probability, read them and create some human-friendly
names for the topics.
"""

# Vi vælger den bedste model og bruger den til at gruppere udtalelserne i emner
best_model_n = model_tests["AntalEmner"].iloc[0]
coh_score, auto_topics, auto_words = fit_lda_model(
    speech_tokens, "Token", "UdtalelseId", best_model_n
)

# For hvert emne finder vi de 5 udtalelser med højest præcision
best_fitting = auto_topics.copy()
best_fitting = best_fitting.sort_values(
    by=["Emne", "Sandsynlighed"], ascending=[True, False]
)
best_fitting["ScoreRank"] = best_fitting.groupby("Emne").cumcount() + 1
best_fitting = best_fitting[best_fitting["ScoreRank"] <= 5].copy()
best_fitting = pd.merge(
    best_fitting,
    all_debates[["UdtalelseId", "Udtalelse"]],
    how="left",
    on="UdtalelseId",
)
# best_fitting.to_excel("output/best_fitting.xlsx", index=False)

# Vi tilføjer vores "menneskevenlige" navne for de enkelte emner
auto_topics = pd.merge(auto_topics, mapping_topics, how="left", on="EmneNr")

print("Gruppering af udtalelser i diverse temaer færdig.")


# %% Manuelle tjek for, at visse ord bliver nævnt

"""
Her unersøger vi blot om nogle specifikke ord bliver nævnt ifm. hver
eneste udtalelse. Det er fx referencer til kriminalitet, udlændingenes
bidrag til/integration i det danske samfund osv.
"""

manual_topics = speech_tokens.copy()

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

# Tjek for, om ovennævnte ord nævnes i de individuelle udtalelser
for var_name, var_val in zip(topics_to_check.keys(), topics_to_check.values()):
    manual_topics[var_name] = manual_topics["Token"].str.contains(var_val)

# Vi konverterer datane fra "bredt" til "langt" format
manual_topics = pd.melt(
    manual_topics,
    id_vars="UdtalelseId",
    value_vars=manual_topics.columns[2:],
    value_name="ReferererTilEmne",
    var_name="Emne",
)
manual_topics["ReferererTilEmne"] = manual_topics["ReferererTilEmne"].astype(int)

print("Manuelle tjek for, at visse ord bliver nævnt færdig.")


# %% Eksport af de færdige analyser

auto_topics.to_parquet("output/results_topics_auto.parquet")
model_tests.to_parquet("output/results_topics_model_stats.parquet")
manual_topics.to_parquet("output/results_topics_manual.parquet")
print("Resultater fra tekstanalyse er eksporteret til 'output data' mappen.")
print("FÆRDIG.")

# %%
