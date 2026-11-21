"""
make_dataset.py
Carga el dataset original y realiza validaciones básicas.
"""
import pandas as pd
import sys
from pathlib import Path

# Añadir la raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_PATH, PROCESSED_DATA_DIR, TARGET
def load_raw_data(path=RAW_DATA_PATH):
    print(f"Cargando datos desde: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nDistribución del target ({TARGET}):")
    print(df[TARGET].value_counts(normalize=True).round(3))
    print(f"\nValores nulos totales: {df.isnull().sum().sum()}")
    return df
def save_processed(df, name="diabetes_clean.csv"):
    """Guarda una versión limpia intermedia."""
    out_path = PROCESSED_DATA_DIR / name
    df.to_csv(out_path, index=False)
    print(f"Guardado en: {out_path}")
    return out_path
if __name__ == "__main__":
    df = load_raw_data()
    save_processed(df)
    print("\n✓ make_dataset.py ejecutado correctamente")