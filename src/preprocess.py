import re
import pandas as pd

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s\|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_soup(df: pd.DataFrame, fields=None):
    if fields is None:
        fields = ['genres','cast','director','description']
    for f in fields:
        df[f] = df[f].fillna("").map(clean_text)
    df['soup'] = df[fields].apply(lambda row: " ".join([str(x) for x in row if x]), axis=1)
    return df
