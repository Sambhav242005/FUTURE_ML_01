from pathlib import Path
import pandas as pd
import kagglehub
from typing import Dict, Hashable, Iterable, List, Tuple

def load_sales_dataset(
    local_dir: str | Path | None = None,
    *,
    filename: str = "sales_data_sample.csv",
    force_download: bool = False,
    return_path: bool = False,
    encoding: str | None = None,          # <-- new
) -> pd.DataFrame | tuple[pd.DataFrame, Path]:
    """
    Fetch kyanyoga/sample-sales-data and return it as a pandas DataFrame.
    If `encoding` is None, tries UTF-8 first, then cp1252 automatically.
    """
    # ---------- resolve download location ----------
    if local_dir is not None:
        local_dir = Path(local_dir).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)

        kagglehub.dataset_download(
            "kyanyoga/sample-sales-data",
            force_download=force_download,
            path=str(local_dir),
        )
        csv_path = local_dir / filename
    else:
        dataset_root = kagglehub.dataset_download(
            "kyanyoga/sample-sales-data",
            force_download=force_download,
        )
        csv_path = Path(dataset_root) / filename

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    # ---------- robust CSV read ----------
    encodings_to_try = [encoding] if encoding else ["utf-8", "cp1252"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:                                      # ran out of options
        raise last_err or UnicodeDecodeError(
            "All encoding attempts failed. "
            "Pass `encoding='...'` explicitly."
        )

    return (df, csv_path) if return_path else df


def assign_numbers(
    items: Iterable[Hashable]
) -> Tuple[List[int], Dict[Hashable, int]]:
    """
    Map each unique item to an integer and return both the mapped list
    and the dict that did the mapping.

    Parameters
    ----------
    items : iterable of hashable objects
        Strings, ints, tuples—anything hashable.  Order doesn't matter.

    Returns
    -------
    encoded : list[int]
        Numeric codes in the same order as `items`.
    mapping : dict
        {original_item: code}.  Useful for decoding later.
    """
    # 1️⃣ build the mapping
    mapping: Dict[Hashable, int] = {}
    next_id = 0
    for obj in items:
        if obj not in mapping:
            mapping[obj] = next_id
            next_id += 1

    # 2️⃣ convert the list with the mapping
    encoded = [mapping[obj] for obj in items]

    return encoded, mapping
