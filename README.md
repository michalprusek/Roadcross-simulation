# Simulace křižovatky


# Úvod

Tato seminární práce se zaměřuje na simulaci dopravní křižovatky typu T pomocí Inteligentního řidičského modelu (IDM). Cílem simulace je nalézt vhodná počáteční rozdělení rychlostí a časových rozestupů mezi vozidly (při generování vozidel na začátku silnice) spolu s optimalizovanými parametry IDM, aby výsledné rozdělení časových intervalů průjezdů vozidel křižovatkou odpovídalo reálným datům získaným na mnichovské křižovatce. Reálná data z této křižovatky mám od profesora Krbálka. Tato data obsahují jak časové intervaly mezi průjezdy dvou po sobě jedoucích aut na hlavní silnici křižovatkou. Dále obsahují druhý sloupec, který říká, kolik aut z vedlejší silnice se do vzniklé mezery na hlavní silnici zařadilo. Dalším cílem je porovnat simulované počty zařazených vozidel do mezery mezi auty na hlavní silnici s reálnými počty zařazených aut.

## straight_line.py

Tento kód simuluje provoz na silnici pomocí modelu **IDM (Intelligent Driver Model)** a generování časových intervalů mezi vozidly dle **GIG (Generalized Inverse Gaussian)** a **log-normálního** rozdělení. Právě tato dvě rozdělení jsem zvolil, protože by měla odpovídat reálným situacím na pozemní komunikaci (často se předpokládá, že právě GIG rozdělení odpovídá rozdělení časových intervalů mezi jednotlivými průjezdy aut). Těmto dvoum rozdělením jsem na začátku nastavil i realistické parametry (hledat parametry IDM a i těchto dvou rozdělení by byl složitější a výpočetně náročnější úkol). Parametry GIG jsem volil na základě nafitovaného GIG rozdělení na reálná data z mnihovské křižovatky. Tyto parametry jsou **lambda = 2.71**, **beta = 1.03**, **loc = 0**. Parametry log-normálního rozdělení jsou nastaveny tak, aby byla střední hodnota rozdělení 50km/h a směrodatná odchylka 5.4km/h. Celý proces pak sbírá a vyhodnocuje data, jako např. časy mezi průjezdy (roadcross) či intervaly pro spawn nových aut - to pak slouží k optimalizaci IDM parametrů.

## Jak to zhruba funguje:

1. **Spuštění simulace**  
   - Načte se konfigurace (max. počet aut, délka silnice, parametry rozdělení atd.).
   - Vytvoří se instance simulace, připraví se okno (pokud je vizualizace zapnutá).

2. **Generování nových vozidel**  
   - V pravidelných intervalech (vzorkovaných z GIG rozdělení) se „spawnuje“ nové vozidlo.
   - Každé vozidlo získá **počáteční rychlost** z log-normálního rozdělení, aby měla auta různou vstupní rychlost.

3. **Aktualizace polohy a rychlosti vozidel**  
   - V každém simulačním kroku se vyhodnocuje akcelerace a brzdění na základě **IDM**, což zohledňuje:
     - Bezpečnostní rozestup od vozidla vepředu.
     - Omezení maximálního zrychlení a zpomalení.
   - Vozidla se posouvají po silnici, pokud nějaké přejede definovaný měřící bod (roadcross), dojde k zaznamenání jeho času.

4. **Ukončení simulace**  
   - Kód má dvě možné ukončení:
     - Buď dojedou všechna vozidla (nikdo už nezůstane před silničním bodem),
     - Nebo by došlo ke kolizi (kód umí tento scénář zachytit) - při rozumném nastavení parametrů rozdělení spawnů a parametrů IDM tento scénář nenastane.

5. **Vyhodnocení výsledků**  
   - Po skončení simulace (pokud není silent režim) proběhne:
     - Vykreslení grafů a histogramů pro časy mezi spawny vozidel, průjezdy měřícím bodem a počáteční rychlosti.
     - Porovnání s reálnými daty (dají se načíst z Excel souboru).
     - **Kolmogorov-Smirnov** testy pro kontrolu shody empirických dat s GIG/log-normální hypotézou. V případě KS testu reálných dat s GIG hypotézou dojde k jeho zamítnutí ačkoliv na základě       vizuálního porovnání fitované GIG rozdělení velmi dobře aproximuje histogram. To si vysvětluji velkým množstvím dat (cca 23000 vzorků).

## Proces hledání optimálních parametrů IDM

**Co jednotlivé parametry znamenají?**

- **min_gap:**
  - Toto je minimální vzdálenost (v metrech), kterou chce řidič udržovat od vozidla před sebou, i když obě auta stojí. Odpovídá „osobnímu prostoru“, který řidič nechává při nízkých rychlostech nebo na semaforech.

- **max_acc:**
  - Maximální možné zrychlení (v metrech za sekundu na druhou), které může vozidlo dosáhnout nebo které je řidič ochoten využít. Reprezentuje, jak agresivně vozidlo dokáže zrychlovat v bezpečných podmínkách.

- **max_dec:**
  - Maximální (komfortní) zpomalení (v metrech za sekundu na druhou), které řidič nebo vozidlo použije. Odpovídá tomu, jak prudce může vozidlo brzdit, aniž by to bylo nekomfortní nebo nebezpečné.

- **react_time:**
  - Časová prodleva (v sekundách), během které řidič reaguje na změny v provozu (např. zpomalení vozidla před ním). Vyšší hodnota znamená, že řidič reaguje pomaleji.

- **desired_speed:**
  - Cílová cestovní rychlost (v metrech za sekundu), které chce řidič dosáhnout za optimálních podmínek. Reprezentuje rychlost, kterou by řidič preferoval na volné silnici.

- **delta:**
  - Tento exponent ovlivňuje, jak rychle se mění akcelerace vozidla v závislosti na vzdálenosti a relativní rychlosti. V modelu IDM bývá hodnota kolem 4. Vyšší hodnota znamená, že model je citlivější na změny ve vzdálenosti a rychlosti, což může vést k agresivnějším nebo naopak opatrnějším reakcím.


Výše popsaný straight_line.py pak slouží k jednotlivým simulacím, které jsou volány v kódu **config_parameters.py**. Tento kód pomocí Grid search algoritmu hledá optimální parametry IDM (**min_gap**,**max_acc**,**max_dec**,**react_time**,**desired_speed**,**delta**) s cílem minimalizovat metriku chi-kvadrát (mezi dvěma histogramy) - metrika podobnosti dvou rozdělení. Cílem je aby při daném počátečním rozdělení rychlostí, intervalů spawnu aut a optimalizovanými parametry IDM bylo rozdělení intervalů průjezdů aut na konci silnice (po 300 metrech) stejné, jako bylo naměřeno na mnichovské křižovatce. Tímto způsobem jsem vygeneroval závislost chi-kvadrát metriky na různých (do mřížky uspořádaných) volbách parametrů IDM. 


