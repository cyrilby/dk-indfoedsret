[![MindGraph](https://img.shields.io/badge/Product%20of%20-MindGraph.dk-1ea2b5?logo=https://mindgraph.dk/favicon.png")](https://mindgraph.dk)

# Indsigter om dansk indfødsret

* Lavet af: Kiril (GitHub@cyrilby)
* Hjemmeside: [mindgraph.dk](https://mindgraph.dk)
* Sidste store opdatering: 25-08-2025

Dette projekt indeholder kode, data og interne logikker, som anvendes for at løbende kunne danne indsigter om dansk indfødsret pba. data fra Folketingets og Danmarks Statistiks hjemmesider.

Projektet har til formål at danne et overblik over hvor omfattende debatterne i salen har været, samt hvilke partier, politikere og emner har fyldt mere ind i diskussionerne. Desuden giver rapporten indsigt i hvilke holdninger de enkelte partier og folketingsmedlemmer har når det kommer til indfødsret, samt hvordan folk har stemt ved tredje behandling.

## Requirements

For at kunne køre Python skripterne i dette projekt kræves der `Python 3.11` samt alle pakker fra `requirments.txt` filen. Der anbefales, at en `venv` sættes op til dette formål.

## Datakilder

### Folketingets hjemmeside

Hoveddelen af dataene, anvendt i dette projekt, kommer fra Folketingets egen [hjemmeside](https://www.ft.dk/), og er tilgængelige offentligt. Dette gælder både data på behandlinger og afstemnigner vedr. de enkelte lovforslag om indfødsret.

Dataene er oprindeligt hentet ved at gemme hver relevant side som en lokal `*.html` fil, som bagefter er indlæst i Python og **omdannet til tabeldata** ved brug af `beautifulsoup4`. Opdatering af disse data (som finder sted 1-2 pr. år) kræver derfor nogle manuelle inputs. Bemærk venligst, at et lovforslag behandles typisk 3 gange inden vedtagelse, og at data på en afstmening fra 2020 mangler pga. en særlig procedure, som Folketinget anvendte under starten af pandemien.

For at se **samtlige links til de sider**, som er blevet behandlet på denne måde, [tryk her](https://github.com/cyrilby/dk-indfoedsret/raw/refs/heads/main/input/lovforslag_oversigt.xlsx).

#### Note vedr. de lokale HTML-filer

Bemærk venligst, at HTML-filerne er manuelt downloaded fra www.ft.dk hjemmesiden, og opbevaret under `input/afstemninger` og `input/behandlinger` mapperne. Dette er fordi Folketingets hjemmeside har indbygget sikring mod web scraping og også fordi lokale kopier giver muligheden for at genproducere dataene ved behov.

Bemærk også, at kun nogle af filerne er uploadet til GitHub fordi sidstnævnte har nogle begrænsinger ift. hvor mange filer og af hvilken størrelse, der må uploades. Hvis du er interesseret i se alle de oprindelige filer, henvend dig til mig eller brug `lovforslag_oversigt.xlsx` filen, som indeholder links til de oprindelige hjemmesider. Filen findes under `input` mappen.

#### Note vedr. output data

Alle output data, genereret i projektet, findes i `output` mappen. Dette gælder både rensede data på behandlinger og afstmeninger og rensede data fra Danmarks Statistik, samt egne beregninger og resultater, dannet af projektets skripter.

### Danmarks Statistik

Data fra Folketinget er suppleret med mere generelle indvandring- og befolkningsdata fra [Danmarks Statistik](www.dst.dk), specifikt nedenstående tabeller:

* `FOLK2`: indeholder generelle befolkningstal, herunder antal mennesker med bopæl i Danmark fordelt på herkomstland og år
* `DKSTAT`: indeholder antallet af uddelte statsborgerskaber fordelt på herkomstland og år
* `VAN66`: indeholder antallet af opholdstilladelser fordelt på opholdsgrundlag og år

## Formål ved de enkelte skripter

De enkelte Python skripter, som er en del af projektet, er beskrevet nedenfor. For at opdatere alle output data i projektet, brug venligst `app.py` skriptet, som sikrer, at alle nedenstående skripter bliver kørt i den rigtige rækkefølge.

### datarens_befolkning.py

Formålet ved dette skript er at indlæse diverse befolkningsdata hentet fra Danmarks Statistik (DST) og gøre dem klar til brug for visualisering og/eller dataanalyse.

### datarens_behandlinger.py

Formålet ved dette skript er at indlæse data om de folketingsdebatter, som finder sted ved **1., 2. og 3. behandling** i salen, fra diverse HTM(L)-filer, samt strukturere de tekstbaserede data i et format, som senere kan bruges til dataanalayse og/eller visualisering.

### datarens_afsteminger.py

Formålet ved dette skript er at indlæse data om folketingsafstemninger fra HTM(L)-filer, finde den relevante tabel med de personlige stemmer, samt kombinere alle relevante tabeller i et samlet datasæt, som kan bruges til dataanalayse og/eller visualisering.

*OBS:* Der indlæses kun data om de **endelige afstemninger** af de forskellige indfødsretslovforslag, som finder sted ved 3. behandling. Evt. andre afstemninger, forbundet med disse lovforslag, er ikke omfattet af dette projekt.

### datarens_partihoppere.py

Formålet ved dette skript er at lave en hjælpetabel", som indeholder de seneste partigrupper for de såkaldte "partihoppere". Dette gør det nemmere at filtrere på personer i Power BI, og sikrer, at vi ikke optæller den samme politiker som to forskellige personer, fx når vi opsummerer deres holdninger gennem en LLM.

### functions_llm.py

Dette skript indeholder funktioner, som gør det muligt at forbinde til OpenAIs LLM model, og anvende den til forskellige formål, fx opsummering af tekst, oversættelse osv.

### functions_sentiment.py

Dette skript indeholder funktioner, som gør det muligt at hente
en sporgmodel fra HuggingFace, og anvende den til sentiment
analyse uden at det påkræver en forbindelse til nettet.

### tekstanalyse_ord.py

Formålet ved dette skript er at tage de rensede  data fra debatterne, som indeholder diverse taler ("udtalelser"), og bruge dem til klassisk tekstanalyse, herunder, identificering af stopwords m.fl. Alt i den her skript foregår lokalt uden at bruge eksterne API'er.

### tekstanalyse_sentiment.py

Formålet ved dette skript er at anvende en sprogmodel for at klassificere samtlige udtalelser i salen under enten positiv, negativ eller neutral stemning (sentiment). Dette gøres ved brug af en lokal model (uden eksterne API'er), oprindeligt hentet fra HuggingFace.

### tekstanalyse_emner.py

Formålet ved dette skript er at tage de rensede data fra debatterne, som indeholder diverse taler ("udtalelser"), og bruge dem til klassisk tekstanalyse, specifikt automatisk gruppering af udtalelserne i diverse kategorier. Alt i den her skript foregår lokalt uden at bruge eksterne API'er. Der anvendes en LDA model til klassificeringen af teksterne.

### tekstanalyse_llm.py

I dette skript anvender vi OpenAIs sprogmodel til forskellige slags foremål, primært opsummering af lange udtalelser og gruppering af udtalelser i nogle kategorier (emner). Pointen ved denne analyse er at supplere de andre emneanalyser, som bruger klassiske værktøjer, og skabe nogle indsigter, som er mere letforståelige end fx LDA.