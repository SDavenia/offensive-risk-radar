
import pandas as pd
import re
TAXONOMY = {
    "arts, culture, entertainment and media": [
        "arts and entertainment", "culture", "mass media",
    ],
    "crime, law and justice": [
        "crime", "judiciary", "justice", "law", "law enforcement",
    ],
    "disaster, accident and emergency incident": [
        "accident and emergency incident", "disaster", "emergency incident",
        "emergency planning", "emergency response",
    ],
    "economy, business and finance": [
        "business information", "products and services", "economy",
        "business enterprise", "market and exchange",
    ],
    "education": [
        "parents group", "religious education", "school", "social learning",
        "teaching and learning", "curriculum",
        "educational testing and examinations", "entrance examination",
        "students", "teachers", "vocational education", "educational grading",
        "online and remote learning",
    ],
    "environment": [
        "climate change", "conservation", "environmental pollution",
        "natural resource", "nature", "sustainability",
    ],
    "health": [
        "disease and condition", "health facility", "health organisation",
        "health treatment and procedure", "government health care",
        "health insurance", "private health care", "medical profession",
        "non-human diseases", "public health",
    ],
    "human interest": [
        "accomplishment", "award and prize", "record and achievement",
        "ceremony", "people", "human mishap", "high society", "celebrity",
        "anniversary", "birthday",
    ],
    "labour": [
        "employment", "employment legislation", "labour market",
        "labour relations", "retirement", "unemployment", "unions",
    ],
    "lifestyle and leisure": [
        "leisure", "lifestyle", "wellness",
    ],
    "politics and government": [
        "election", "government", "government policy", "international relations",
        "non-governmental organisation (NGO)", "political crisis",
        "political prisoners and dissenters", "political process",
    ],
    "religion": [
        "belief systems", "interreligious dialogue", "religious conflict",
        "religious event", "religious festival and holiday", "religious ritual",
        "religious facility", "relations between religion and government",
        "religious leader", "religious text",
    ],
    "science and technology": [
        "biomedical science", "mathematics", "natural science",
        "scientific research", "scientific institution", "social sciences",
        "scientific standards", "technology and engineering",
    ],
    "society": [
        "fundamental rights", "communities", "demographics", "immigration",
        "emigration", "discrimination", "family", "demographic group",
        "social condition", "social problem", "values", "welfare",
        "diversity, equity and inclusion",
    ],
    "sport": [
        "competition discipline", "disciplinary action in sport",
        "drug use in sport", "sport event", "sport industry",
        "sport organisation", "sport venue", "sports transaction",
        "sport achievement", "sports coaching",
        "sports management and ownership", "sports officiating",
    ],
    "conflict, war and peace": [
        "act of terror", "armed conflict", "civil unrest", "coup d'etat",
        "massacre", "peace process", "post-war reconstruction", "cyber warfare",
        "war victims",
    ],
    "weather": [
        "weather forecast", "weather phenomena", "weather statistic",
        "weather warning",
    ],
}


def _format_taxonomy(taxonomy: dict) -> str:
    """Human-readable category list for the prompt (instances shown only as hints)."""
    lines = []
    for cat, instances in taxonomy.items():
        if instances:
            lines.append(f"- {cat} (es.: {', '.join(instances)})")
        else:
            lines.append(f"- {cat}")
    return "\n".join(lines)


def _norm(s: str) -> str:
    """Lower, collapse whitespace, normalise apostrophes — for robust matching."""
    s = str(s).strip().lower().replace("\u2019", "'")
    return re.sub(r"\s+", " ", s)


# Category names take precedence over instance names on any ambiguity.
_CAT_BY_NORM = {_norm(c): c for c in TAXONOMY}
_CATEGORY_KEYS = sorted(_CAT_BY_NORM, key=len, reverse=True)

_INSTANCE_LOOKUP: dict = {}
for _cat, _insts in TAXONOMY.items():
    for _inst in _insts:
        _INSTANCE_LOOKUP.setdefault(_norm(_inst), _cat)
_INSTANCE_KEYS = sorted(_INSTANCE_LOOKUP, key=len, reverse=True)

def parse_topic(raw: str):
    """Map a model output to one of the 17 categories. None if nothing matches.

    Order: exact category -> exact instance -> category substring (longest first)
    -> instance substring (longest first). Categories always win over instances
    so a clean 'society' is never shadowed by a longer instance string."""
    if not raw:
        return None
    text = _norm(raw)

    if text in _CAT_BY_NORM:
        return _CAT_BY_NORM[text]
    if text in _INSTANCE_LOOKUP:
        return _INSTANCE_LOOKUP[text]
    for key in _CATEGORY_KEYS:
        if key in text:
            return _CAT_BY_NORM[key]
    for key in _INSTANCE_KEYS:
        if key in text:
            return _INSTANCE_LOOKUP[key]
    return None



def canonical_topic(raw):
    """Normalise a GOLD topic value to its top-level category.
    Accepts a category, an instance, or (defensively) a 1-element list.
    Unknown strings are returned as-is so they surface as their own class."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
        if raw is None:
            return None
    text = _norm(raw)
    if text in _CAT_BY_NORM:
        return _CAT_BY_NORM[text]
    if text in _INSTANCE_LOOKUP:
        return _INSTANCE_LOOKUP[text]
    return str(raw)
