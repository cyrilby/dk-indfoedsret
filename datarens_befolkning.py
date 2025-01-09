"""
=================================
Import og rens af befolkningsdata
=================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 09-01-2025

Formålet ved dette skript er at indlæse diverse befolkningsdata hentet
fra Danmarks Statistik (DST) og gøre dem klar til brug for visualisering
og/eller dataanalyse.

=====================================
NÆSTE OPDATERING AF DATAENE I KILDEN:
=====================================
DKSTAT: Opdateres næste gang: 12-02-2025 08:00 med perioden 2024
FOLK2: Opdateres næste gang: 12-02-2025 08:00 med perioden 2025
VAN66: Opdateres næste gang: 19-02-2025 08:00 med perioden 2024
"""

# %% Generel opsætning

# Import af relevante pakker
import pandas as pd


# %% Egen funktion til indlæsning af DST data


def read_dst_data(
    lokal_fil: str, id_cols: list[str], melt_df: bool = True, var_navn: str = "Antal"
) -> pd.DataFrame:
    """
    Indlæser et datasæt fra Danmarks Statistik i Excel format,
    samt forbereder dataene til brug i andre analyser.

    Args:
        lokal_fil (str): lokal Excel fil til indlæsning
        id_cols (list[str]): hvad de første X kolonner, som indeholder
        ID-oplysninger, skal hedde
        melt_df (bool, optional): skal datasættet konverteres fra "bredt"
        til "langt" format. Som udgangspunkt "True".
        lokal_fil (str): hvad kolonnen, som skabes ved ovennævnte procedure,
        skal hedde. Som udgangspunkt "Antal".

    Returns:
        pd.DataFrame: renset dataset klar til videre brug
    """

    # Vi indlæser data og omdøber de første tre kolonner
    temp_data = pd.read_excel(lokal_fil, skiprows=2)
    cols_with_years = temp_data.columns[len(id_cols) :].tolist()
    temp_data.columns = id_cols + cols_with_years

    # DST filer indeholder ikke gentagne værdier i rækkerne, så vi udfylder de blanke felter
    for col in id_cols:
        temp_data[col] = temp_data[col].ffill()

    # DST filer indeholder år i kolonner, men disse skal fremstå i rækker i stedet for
    if melt_df:
        temp_data = pd.melt(
            temp_data,
            id_vars=id_cols,
            value_vars=cols_with_years,
            var_name="År",
            value_name=var_navn,
        )
    return temp_data


# %% Egen funktion, som tjekker for manglende værdier


def check_missing(input_df: pd.DataFrame, df_name: str, pct_digits: int = 1) -> None:
    """
    Tjekker alle kolonner i et datasæt for manglende værdier.
    Udskriver en opsummering for enhver kolonne i datasættet.

    Args:
        input_df (pd.DataFrame): datasæt at tjekke for mangler
        df_name (str): deskriptivt navn af datasættet
        pct_digits (int, optional): hvor mange tal, der skal vises
        efter kommaen, når opsummering udskrives. Som udgangspunkt 1.
    """

    print(f"Tjek for manglende værdier i '{df_name}' tabellen:")
    for col in input_df.columns:
        n_missing = input_df[col].isna().sum()
        pct_missing = n_missing / len(input_df)
        pct_missing = round(100 * pct_missing, pct_digits)
        print(f"'{col}': {pct_missing}% manglende værdier.")


# %% Rens af generelle befolkningstal (FOLK2)

table_name = "FOLK2"
print(f"Import og rens af data fra '{table_name}' tabellen er nu i gang...\n")

# Vi indlæser data og omdøber de første X kolonner
id_cols = ["Statsborgerskab", "Herkomst", "Herkomstland"]
general_population = read_dst_data(
    "input/dst/FOLK2.xlsx", id_cols, True, "AntalMennesker"
)

# Vi tjekker for mangler i data
check_missing(general_population, table_name)

print(f"\nTabellen '{table_name}' er nu klar til brug.")


# %% Rens af tal om erhvervelse af statsborgerskab (DKSTAT)

table_name = "DKSTAT"
print(f"Import og rens af data fra '{table_name}' tabellen er nu i gang...\n")

# Vi indlæser data og omdøber de første X kolonner
id_cols = ["Herkomstland"]
citizenship_acquisition = read_dst_data(
    "input/dst/DKSTAT.xlsx", id_cols, True, "AntalMennesker"
)

# Vi tjekker for mangler i data
check_missing(citizenship_acquisition, table_name)

# Vi erstatter manglende værdier i "AntalMennesker" med 0
citizenship_acquisition["AntalMennesker"] = citizenship_acquisition[
    "AntalMennesker"
].fillna(0)
citizenship_acquisition["AntalMennesker"] = citizenship_acquisition[
    "AntalMennesker"
].astype(int)
print("OBS: Manglende værdier i 'AntalMennesker' kolonnen erstattet med '0'.")

# Vi dropper fodnoter fra tabellen
footnote = "Tallene for årene 2007-2015 er den 13. februar 2017 revideret"
citizenship_acquisition = citizenship_acquisition[
    ~citizenship_acquisition["Herkomstland"].str.contains(footnote)
].copy()
citizenship_acquisition = citizenship_acquisition.reset_index(drop=True)

print(f"\nTabellen '{table_name}' er nu klar til brug.")


# %% Rens af tal om opholdstilladelser (VAN66)

# Dataene her er adskilt i 2 fordi DST kan kun eksportere op til 100K rækker i 1 fil
table_name = "VAN66"
print(f"Import og rens af data fra '{table_name}' tabellen er nu i gang...\n")

# Vi importerer data fra den 1. fil
id_cols = ["Opholdsgrundlag", "Herkomstland"]
residence_permits_1 = read_dst_data(
    "input/dst/VAN66 - asyl, familiesammenhøring og øvrige.xlsx",
    id_cols,
    True,
    "AntalMennesker",
)

# Vi importerer data fra den 2. fil
id_cols = ["Opholdsgrundlag", "Herkomstland"]
residence_permits_2 = read_dst_data(
    "input/dst/VAN66 - studie, erhverv, EU, EØS.xlsx", id_cols, True, "AntalMennesker"
)

# Vi samler alle dataene i et datasæt
residence_permits = pd.concat([residence_permits_1, residence_permits_2])

# Vi tjekker for mangler i data
check_missing(residence_permits, table_name)

# Vi erstatter manglende værdier i "AntalMennesker" med 0
residence_permits["AntalMennesker"] = residence_permits["AntalMennesker"].fillna(0)
residence_permits["AntalMennesker"] = residence_permits["AntalMennesker"].astype(int)
print("OBS: Manglende værdier i 'AntalMennesker' kolonnen erstattet med '0'.")

# Vi dropper fodnoter fra tabellen
footnote = "Antallet af opholdstilladelser i 2006-2011 er opjusteret"
residence_permits = residence_permits[
    ~residence_permits["Opholdsgrundlag"].str.contains(footnote)
].copy()
residence_permits = residence_permits.reset_index(drop=True)

print(f"\nTabellen '{table_name}' er nu klar til brug.")


# %% Eksport af de rensede data

general_population.to_parquet("output/dst_befolkning.parquet")
citizenship_acquisition.to_parquet("output/dst_statsborgerskab.parquet")
residence_permits.to_parquet("output/dst_opholdstilladelser.parquet")
print("\nRensede data vedr. befolkningstal er nu klar til brug i 'output' mappen.")
print("FÆRDIG.")

# %%
