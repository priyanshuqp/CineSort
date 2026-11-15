"""Clean Wikipedia-style movie plot CSVs the same way as Prototype.ipynb."""

from __future__ import annotations

import pandas as pd

DEFAULT_TEXT_COL = "Plot"
DEFAULT_GENRE_COL = "Genre"
TOP_N_GENRES = 6
DROP_LABELS = {"unknown", "", "nan"}


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    text = cols.get("plot") or cols.get("text") or cols.get("synopsis")
    genre = cols.get("genre") or cols.get("genres") or cols.get("genre_raw")
    if text is None or genre is None:
        raise ValueError(
            "CSV needs a plot/text column and a genre column "
            f"(found: {list(df.columns)})"
        )
    return text, genre


def clean_movies(
    df: pd.DataFrame,
    text_col: str | None = None,
    genre_col: str | None = None,
    top_n: int = TOP_N_GENRES,
) -> pd.DataFrame:
    if text_col is None or genre_col is None:
        text_col, genre_col = detect_columns(df)

    out = df[[text_col, genre_col]].dropna()
    out.columns = ["text", "genre_raw"]
    out["genre"] = (
        out["genre_raw"]
        .astype(str)
        .str.lower()
        .str.split(r"[|,/]", regex=True)
        .str[0]
        .str.strip()
    )
    out = out[~out["genre"].isin(DROP_LABELS)]
    top = out["genre"].value_counts().head(top_n).index
    out = out[out["genre"].isin(top)].reset_index(drop=True)
    return out[["text", "genre"]]
