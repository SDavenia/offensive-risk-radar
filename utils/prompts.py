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

