# Indsigter om dansk indfødsret

* Lavet af: kirilboyanovbg[at]gmail.com
* Sidste mærkbare opdatering: 11-09-2024

Dette projekt er igangværende pr. 11-09-2024, og handler om at danne nogle indsigter omkring dansk indfødsret, herunder hvilket sprog, der bliver brugt af de danske politiker, samt at undersøge om de debatter, som holdes i Folketinget, afspejler og/eller påvirker befolkningsstatistikkerne.

De enkelte Python skripter, som er en del af projektet, er beskrevet nedenfor.

## datarens_befolkning.py

Formålet ved dette skript er at indlæse diverse befolkningsdata hentet fra Danmarks Statistik (DST) og gøre dem klar til brug for visualisering og/eller dataanalyse.

Specifikt bruges der **følgende tabeller** fra www.dst.dk:

* `FOLK02`: indeholder generelle befolkningstal, herunder antal mennesker med bopæl i Danmark fordelt på herkomstland og år
* `DKSTAT`: indeholder antallet af uddelte statsborgerskaber fordelt på herkomstland og år
* `VAN66`: indeholder antallet af opholdstilladelser fordelt på opholdsgrundlag og år

## datarens_afsteminger.py

Formålet ved dette skript er at indlæse data om folketingsafstemninger fra HTM(L)-filer, finde den relevante tabel med de personlige stemmer, samt kombinere alle relevante tabeller i et samlet datasæt, som kan bruges til dataanalayse og/eller visualisering.

*OBS:* Der indlæses kun data om de **endelige afstemninger** af de forskellige indfødsretslovforslag, som finder sted ved 3. behandling. Evt. andre afstemninger, forbundet med disse lovforslag, er ikke omfattet af dette projekt.

## datarens_behandlinger.py

Formålet ved dette skript er at indlæse data om de folketingsdebatter, som finder sted ved **1., 2. og 3. behandling** i salen, fra diverse HTM(L)-filer, samt strukturere de tekstbaserede data i et format, som senere kan bruges til dataanalayse og/eller visualisering.

## Note vedr. lokale HTML-filer

Bemærk venligst, at HTML-filerne er manuelt downloaded fra www.ft.dk hjemmesiden, og opbevaret under `input/afstemninger` og `input/behandlinger` mapperne. Dette er fordi Folketingets hjemmeside har indbygget sikring mod web scraping og også fordi lokale kopier giver muligheden for at genproducere dataene ved behov.