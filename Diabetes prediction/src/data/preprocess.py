"""
preprocess.py
Pipeline de preprocesamiento: split, escalado y guardado.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
 RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
 TARGET, RANDOM_STATE, TEST_SIZE
)
def get_feature_lists(df):
 """Separa features numéricas y el resto."""
 numeric_features = ["BMI", "MentHlth", "PhysHlth"]
 # El resto se tratan como categóricas/ordinales (ya codificadas)
 other_features = [c for c in df.columns if c not in numeric_features + [TARGET]]
 return numeric_features, other_features

def preprocess_and_split(df=None):
 """
 - Carga datos si no se pasan
 - Separa X e y
 - Train/Test split estratificado
 - Escala solo variables numéricas
 - Guarda todo en data/processed/
 """
 if df is None:
   df = pd.read_csv(RAW_DATA_PATH)
 X = df.drop(columns=[TARGET])
 y = df[TARGET]
 X_train, X_test, y_train, y_test = train_test_split(
 X, y,
 test_size=TEST_SIZE,
 random_state=RANDOM_STATE,
 stratify=y
 )
 numeric_features, _ = get_feature_lists(df)
 # Pipeline de escalado solo para numéricas
 preprocessor = ColumnTransformer(
 transformers=[
 ("num", StandardScaler(), numeric_features)
 ],
 remainder="passthrough" # deja las demás sin cambios
 )
 X_train_processed = preprocessor.fit_transform(X_train)
 X_test_processed = preprocessor.transform(X_test)
 # Recuperar nombres de columnas
 feature_names = numeric_features + [c for c in X.columns if c not in numeric_features]
 X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
 X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
 # Guardar
 X_train_df.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
 X_test_df.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
 y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
 y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
 # Guardar el preprocessor para usarlo en predicción
 joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
 print(f"Train shape: {X_train_df.shape} | Test shape: {X_test_df.shape}")
 print(f"Distribución y_train:\n{y_train.value_counts(normalize=True).round(3)}")
 print("✓ Preprocesamiento completado y archivos guardados")
 return X_train_df, X_test_df, y_train, y_test, preprocessor
if __name__ == "__main__":
 preprocess_and_split()
 