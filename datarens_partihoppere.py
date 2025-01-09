"""
==========================
Håndtering af partihoppere
==========================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 09-01-2025

Formålet ved dette skript er at lave  en "hjælpetabel", som indeholder
de seneste partigrupper for de såkaldte "partihoppere". Dette vil gøre
det nemmere at filtrere på personer i Power BI, og vil sikre, at vi
ikke optæller den samme politiker som to forskellige personer, fx
når vi opsummerer deres holdninger gennem OpenAIs LLM.
"""

# %% Generel opsætning

# Import af relevante pakker
import pandas as pd
import numpy as np

# Import af de nyeste data på afstemninger
final_votes = pd.read_parquet("output/ft_afstemninger.parquet")

# Import af allerede bekræftede partihoppere
relevant_cols = ["År", "Sæson", "Navn", "Status"]
confirmed_hoppers = pd.read_excel(
    "input/mapping_tabeller.xlsx", sheet_name="Partihoppere"
)[relevant_cols]


# %% Automatisk håndtering af partihoppere

"""
For at kunne filtrere på enkelte folketingsmedlemmer samt
for at kunne opsummere deres holdning er det nødvendigt, at
hver person, der har været medlem af flere partigrupper,
bliver optalt som den samme person.
"""

print("Automatisk håndtering af partihoppere er nu i gang...")

# Vi tilføjer en talkolonne til brug i sortering
party_membership = final_votes.copy()
party_membership["Tid"] = party_membership["Sæson"].map({"Forår": 1, "Efterår": 2})
party_membership["Tid"] = party_membership["År"].astype(str) + party_membership[
    "Tid"
].astype(str)
party_membership["Tid"] = party_membership["Tid"].astype(int)

# # Inden vi fortsætter, tjekker vi, om der er personer med
# # identiske navne i løbet af den samme sæson
# vars_for_group = ["År", "Sæson", "Navn"]
# party_membership["AntalIdentiskeNavne"] = party_membership.groupby(vars_for_group)["Navn"].transform("nunique")

# Vi laver en liste af alle folketingsmedlemmer for at se,
# om der gennem årene har været nogle partihoppere
cols_for_sort = ["Navn", "Tid"]
asc_order_sort = [True, False]
party_membership = party_membership.sort_values(
    by=cols_for_sort, ascending=asc_order_sort
)
party_membership["PartiGruppeKort"] = party_membership["PartiGruppe"].str.extract(
    r"\((.*?)\)"
)
party_membership["AntalPartier"] = party_membership.groupby("Navn")[
    "PartiGruppeKort"
].transform("nunique")
party_membership["PartiHopper"] = party_membership["AntalPartier"] > 1

# For identificerede partihoppere finder vi deres seneste
# (nuværende) partigruppe
party_membership["MaksTid"] = party_membership.groupby("Navn")["Tid"].transform("max")
party_membership["SenestePartiGruppeKort"] = np.where(
    party_membership["Tid"] == party_membership["MaksTid"],
    party_membership["PartiGruppeKort"],
    np.nan,
)
party_membership["SenestePartiGruppe"] = np.where(
    party_membership["Tid"] == party_membership["MaksTid"],
    party_membership["PartiGruppe"],
    np.nan,
)

for col in ["SenestePartiGruppe", "SenestePartiGruppeKort"]:
    party_membership[col] = party_membership.groupby("Navn")[col].transform(
        lambda x: x.ffill().bfill()
    )

# Vi beholder kun nogle kolonner
cols_to_drop = ["Stemme", "Tid", "MaksTid"]
party_membership = party_membership.drop(columns=cols_to_drop)
party_membership = party_membership.reset_index(drop=True)

# Særskilt tabel kun med potentielle partihopper
id_cols = ["Navn", "PartiGruppe", "SenestePartiGruppe"]
party_hoppers = party_membership[party_membership["AntalPartier"] > 1].copy()
party_hoppers = party_hoppers.drop_duplicates(subset=id_cols)
party_hoppers = party_hoppers.reset_index(drop=True)

# Tjek om, der er nye, ubekfræftede partihoppere
id_cols = ["År", "Sæson", "Navn"]
uncomfirmed_hopppers = pd.merge(
    party_hoppers, confirmed_hoppers, how="left", on=id_cols
)
uncomfirmed_hopppers = uncomfirmed_hopppers[
    uncomfirmed_hopppers["Status"].isna()
].copy()
n_hoppers = len(uncomfirmed_hopppers["Navn"].unique())

# Resultater
if n_hoppers:
    print(f"OBS: {n_hoppers} ubekfæftede partihoppere er identificeret.")
    print("Tjek 'output/ft_partihoppere.xlsx' og opdater mapping tabel.")
    party_hoppers.to_excel("output/ft_partihoppere.xlsx", index=False)
else:
    print("Ingen ubekfræftede partihoppere er blevet fundet.")


# %% Manuel bekræftelse af partihoppere

"""
Instruktioner: kør skripter indtil her, opdater mapping tabellen,
og så kør skriptet en gang til (enten forfra eller herfra).
Automatisk fundne partihoppere kan enten have "Bekræftet"
eller "Afvist" i "Status"-kolonnen i mapping tabellen.
"""

# Vi tilføjer status fra det manuelle tjek til dataene
id_cols = ["År", "Sæson", "Navn"]
party_membership = pd.merge(party_membership, confirmed_hoppers, how="left", on=id_cols)

# Hvis vi har "Afvist" status, benytter vi altid nuværende navn
# jf. det, der allerede står i rækken
rejected_hopper = party_membership["Status"] == "Afvist"
party_membership["SenestePartiGruppe"] = np.where(
    rejected_hopper,
    party_membership["PartiGruppe"],
    party_membership["SenestePartiGruppe"],
)
party_membership["SenestePartiGruppeKort"] = np.where(
    rejected_hopper,
    party_membership["PartiGruppeKort"],
    party_membership["SenestePartiGruppeKort"],
)

# Vi tilføjer "Auto" som status for rækker, som mangler eller
# ikke behøver en bekræftelse
party_membership["Status"] = party_membership["Status"].fillna("Auto")


# %% Eksport af de rensede data

party_membership.to_parquet("output/ft_seneste_partigruppe.parquet")
print("Data om partihoppere er nu klar til brug i 'output' mappen.")
print("FÆRDIG.")


# %%
