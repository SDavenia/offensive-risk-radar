ANNOTATION_PROMPT_METADATA = """
Sei un annotatore esperto nella classificazione tematica di articoli di notizie in italiano.

Il tuo compito è assegnare all'ARTICOLO UNA SOLA categoria tematica, scegliendola esclusivamente dalla lista di categorie qui sotto.

CATEGORIE DISPONIBILI

Scegli una sola tra le seguenti categorie. Tra parentesi sono riportati alcuni esempi di sottotemi che appartengono a quella categoria: servono solo a chiarire cosa copre la categoria e NON sono opzioni di risposta valide.

{taxonomy}

TASK DI ANNOTAZIONE

Leggi attentamente il titolo e la descrizione dell'articolo, individua il tema principale e assegna la categoria più pertinente tra quelle elencate sopra.

ISTRUZIONI IMPORTANTI

- Considera l'articolo nel suo complesso e identifica il suo argomento PRINCIPALE.
- Scegli una sola categoria, quella che meglio rappresenta il contenuto.
- Rispondi ESCLUSIVAMENTE con il nome ESATTO di una categoria, copiato letteralmente dalla lista (senza gli esempi tra parentesi).
- Non inventare nuove categorie e non rispondere con i sottotemi tra parentesi.
- Non fornire spiegazioni, punteggiatura extra o testo aggiuntivo.

FORMATO INPUT

ARTICLE TITLE:
{article_title}

ARTICLE DESCRIPTION:
{article_description}

FORMATO OUTPUT

Restituisci SOLO il nome di una categoria dalla lista, ad esempio:

politics and government
"""


ANNOTATION_PROMPT_COMMENTS = """
Sei un annotatore esperto nel riconoscimento di linguaggio offensivo nei commenti online in italiano.

Il tuo compito è determinare se il COMMENTO TARGET potrebbe ragionevolmente essere considerato offensivo da qualcuno.

Devi annotare SOLO il COMMENTO TARGET, utilizzando eventuali informazioni contestuali esclusivamente per interpretarne meglio significato, tono, riferimenti e intenzione comunicativa.

DEFINIZIONE

Un commento è considerato offensivo se contiene linguaggio volgare, insultante, aggressivo, denigratorio, discriminatorio, minaccioso o gravemente irrispettoso, in modo esplicito o implicito, verso una persona o un gruppo.

Il criterio centrale è l’offensività potenziale:
non devi valutare se il commento offende te personalmente, ma se qualcuno potrebbe ragionevolmente percepirlo come offensivo.

TASK DI ANNOTAZIONE

Leggi attentamente tutto il contesto disponibile.

Poi rispondi alla seguente domanda:

"Questo COMMENTO TARGET potrebbe ragionevolmente essere considerato offensivo da qualcuno?"

Restituisci SOLO una delle seguenti etichette:

- Sì
- No

Non fornire spiegazioni o testo aggiuntivo.

ISTRUZIONI IMPORTANTI

- Annota SOLO il COMMENTO TARGET.
- Usa il titolo e la descrizione del video esclusivamente per comprendere il tema, il contesto e i riferimenti del commento.
- Usa eventuali commenti precedenti esclusivamente per comprendere tono, sarcasmo, riferimenti, bersagli o intenzioni.
- Non annotare il titolo del video, la descrizione o i commenti precedenti.
- Considera sia l’offensività esplicita sia quella implicita.
- Valuta insieme tono, intenzione, linguaggio e contesto conversazionale.
- Se un commento contiene sarcasmo, ironia, derisione o ostilità implicita che potrebbe ragionevolmente risultare offensiva, etichettalo come offensivo.
- In caso di dubbio, considera l’effetto complessivo del messaggio piuttosto che singole parole isolate.
- In caso di dubbio, preferisci "Sì" se una persona ragionevole potrebbe percepire il commento come offensivo.

ETICHETTA "Sì" SE IL COMMENTO TARGET:

- Contiene insulti, parolacce, epiteti o linguaggio denigratorio
- Attacca, umilia, deride o svaluta una persona o un gruppo
- Usa linguaggio volgare con intento aggressivo o ostile
- Esprime odio, disprezzo o forte mancanza di rispetto
- Colpisce etnia, nazionalità, genere, religione, orientamento sessuale, disabilità, appartenenza politica o caratteristiche simili
- Contiene minacce, intimidazioni o auguri di danno
- Usa sarcasmo o ironia con effetto offensivo
- Potrebbe essere percepito come offensivo da almeno una persona ragionevole

ETICHETTA "No" SE IL COMMENTO TARGET:

- Esprime disaccordo o critica in modo rispettoso
- Usa linguaggio informale o colloquiale senza aggressività
- Discute temi sensibili in modo neutro o analitico
- Riporta termini offensivi senza approvarli
- Contiene espressioni emotive leggere senza intento offensivo
- È solo vagamente scortese o ambiguo senza offensività chiara

GESTIONE DEL CONTESTO

L’input può contenere:

- VIDEO TITLE: titolo del video
- VIDEO DESCRIPTION: descrizione del video
- HEAD COMMENT: commento principale della conversazione
- PREVIOUS COMMENT: commento immediatamente precedente
- TARGET COMMENT: commento da annotare

Alcuni campi possono essere vuoti se non disponibili.

Regole:

- Usa il contesto solo per interpretare correttamente il COMMENTO TARGET.
- Un commento apparentemente neutro può diventare offensivo nel contesto della conversazione.
- Una parola apparentemente offensiva può essere neutra a seconda del contesto.
- L’etichetta finale deve riferirsi esclusivamente al COMMENTO TARGET.

FORMATO INPUT

VIDEO TITLE:
{video_title}

VIDEO DESCRIPTION:
{video_description}

HEAD COMMENT:
{head_comment}

PREVIOUS COMMENT:
{previous_comment}

TARGET COMMENT:
{target_comment}

FORMATO OUTPUT

Restituisci SOLO:

Sì

oppure

No
"""


OFFENSIVE_RADAR_PROMPTS = [
    """Sei un sistema automatico di moderazione di una piattaforma di notizie online.

Dato il titolo e la descrizione di un video, devi prevedere se la discussione generata attirerà una quantità significativa di commenti offensivi.

Per commenti offensivi si intendono insulti, aggressioni verbali, linguaggio d'odio, discriminazione, denigrazione o attacchi verso persone o gruppi.

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Questo video attirerà una quantità significativa di commenti offensivi?

Rispondi esclusivamente con una sola parola:
"Sì" oppure "No".
""",

    """Sei un esperto di moderazione di contenuti online in italiano.

DEFINIZIONE: un commento è offensivo se contiene linguaggio volgare, insultante, aggressivo, denigratorio, discriminatorio o minaccioso verso una persona o un gruppo.

COMPITO: leggendo SOLO il titolo e la descrizione di un video di notizie, prevedi se il video attirerà un numero rilevante di commenti offensivi da parte degli utenti.

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Considera quanto l'argomento è delicato, divisivo o emotivamente carico.

Rispondi solo con "Sì" (attirerà commenti offensivi) oppure "No" (non li attirerà). Nessun'altra parola.
""",

    """Immagina la sezione commenti di questo video di notizie.

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Valuta se è probabile che la discussione contenga numerosi commenti offensivi, aggressivi, denigratori o ostili verso persone, gruppi o istituzioni.

Non considerare la semplice presenza occasionale di qualche insulto: valuta se l'offensività sarà una componente rilevante della discussione.

Rispondi esclusivamente con:
"Sì" oppure "No".
""",

    """Sei un moderatore professionista di contenuti online.

Il tuo compito è prevedere se un video di notizie genererà una discussione con una quantità rilevante di commenti offensivi.

Valuta mentalmente:
- il livello di conflittualità del tema;
- il potenziale di indignazione pubblica;
- la presenza di gruppi frequentemente bersaglio di ostilità;
- la probabilità di scontri verbali tra utenti.

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Rispondi soltanto con:
"Sì" oppure "No".""",

    """Sei un sistema di previsione del rischio di offensività nelle discussioni online.

Utilizzando esclusivamente il titolo e la descrizione del video, stima se la conversazione che seguirà presenterà un livello elevato di commenti offensivi.

Per livello elevato si intende una presenza consistente di:
- insulti;
- aggressività verbale;
- linguaggio d'odio;
- discriminazione;
- attacchi personali o verso gruppi sociali.

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Output consentito:
"Sì"
oppure
"No"
""",
]