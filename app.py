"""
=============================
Opdatering af projektets data
=============================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 01-08-2025

Dette skript kan bruges til at sikre, at alle data, som kommer
som output fra projektet, er opdateret. Skriptet kalder nemlig
alle andre relevante skripts i den rigtige rækkefølge.
"""

# %% Generel opsætning

import runpy


def run_script(script: str) -> None:
    """
    Kører et andet Python skript gennem det nuværende skript.
    Printer output fra det andet skript i konsolen
    Executes a Python script from within the current script.
    Prints the output of the script in the console.

    Args:
        script (str): skript, som skal køres
    """

    n_chars = len(script) + 22
    print(n_chars * "=")
    print(f"Executing script '{script}'...")
    print(n_chars * "=")
    _ = runpy.run_path(script)
    print("\n\n\n")


# %% Opdatering af data

# Data på debatterne under lovforslagenes behandlinger
run_script("datarens_behandlinger.py")

# Data på enedelige afstemninger af lovforslagene
run_script("datarens_afstemninger.py")

# Identificering og bekræftelse af partihoppere
run_script("datarens_partihoppere.py")

# Befolkningsdata m.fl. fra Danmarks Statistik
run_script("datarens_befolkning.py")

# Tekstanalyse del 1 af 4: tokenization, ordfrekvenser m.fl.
run_script("tekstanalyse_ord.py")

# Tekstanalyse del 2 af 4: sentiment analyse
run_script("tekstanalyse_sentiment.py")

# Tekstanalyse del 3 af 4: emenenalyse gennem LDA model
run_script("tekstanalyse_emner.py")

# Tekstanalyse del 4 af 4: opsummeringer m.fl. gennem ChatGPT
run_script("tekstanalyse_llm.py")

print("Alle data er nu opdateret.")
print("FÆRDIG.")

# %%
