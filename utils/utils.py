import re
import pandas as pd
from sklearn.model_selection import train_test_split

newspapers = [
    "corriere_della_sera",
    "il_gazzettino",
    "ilmessaggero",
    "lastampa",
    "repubblica",
]


def make_dev_test_split(gold_df, test_size: float = 0.4, seed: int = 42):
    """
    60/40 dev/test, stratified on topic 
    """
    strat = gold_df["topic"]
    if strat.value_counts().min() < 2:
        print("WARNING: some topics have < 2 examples — splitting without stratification")
        strat = None
    return train_test_split(gold_df, test_size=test_size, random_state=seed, stratify=strat)


def add_context_columns(df_gold):
    """
    Add head_comment_text and previous_comment_text columns based on depth:
      depth 0 : both None
      depth 1 : head = depth-0 ancestor's text, previous = None
      depth 2+: head = depth-0 ancestor's text, previous = immediate parent's text
    """
    id_to_text = df_gold.set_index("comment_id")["text"].to_dict()
    id_to_parent = df_gold.set_index("comment_id")["inferred_parent_id"].to_dict()

    def get_head_text(row):
        if row["depth"] == 0:
            return None
        cid = row["comment_id"]
        seen = set()
        while cid in id_to_parent and pd.notna(id_to_parent.get(cid)):
            if cid in seen:
                break
            seen.add(cid)
            cid = id_to_parent[cid]
        return id_to_text.get(cid)

    def get_previous_text(row):
        if row["depth"] <= 1:
            return None
        return id_to_text.get(row["inferred_parent_id"])

    df_gold["head_comment_text"] = df_gold.apply(get_head_text, axis=1)
    df_gold["previous_comment_text"] = df_gold.apply(get_previous_text, axis=1)
    return df_gold


_YES = re.compile(r"\bsì\b")
_NO = re.compile(r"\bno\b")
def parse_label(raw: str):
    """Find yes/no in italian and return it. Usable for comments and offensive radar task as they are both binary"""
    text = raw.strip().lower()
    if not text:
        return None

    yes_hits = list(_YES.finditer(text))
    no_hits = list(_NO.finditer(text))
    last_yes = yes_hits[-1].start() if yes_hits else -1
    last_no = no_hits[-1].start() if no_hits else -1

    if last_yes != -1 or last_no != -1:
        return 1 if last_yes > last_no else 0

    # Weak fallback: unaccented "si" only if it is essentially the whole output.
    if re.fullmatch(r"si[.!]?", text):
        return 1
    return None