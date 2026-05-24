import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "dirt_rally_multilabel.csv"
OUTPUT_FILE = "dirt_rally_multilabel_clean.csv"

# -----------------------------
# CARGAR CSV
# -----------------------------
df = pd.read_csv(INPUT_FILE)

print(f"\nFilas originales: {len(df)}")

# -----------------------------
# COLUMNAS MANO DERECHA
# -----------------------------
right_cols = []

for i in range(21):
    right_cols += [f"R_x{i}", f"R_y{i}", f"R_z{i}"]

# -----------------------------
# COLUMNAS MANO IZQUIERDA
# -----------------------------
left_cols = []

for i in range(21):
    left_cols += [f"L_x{i}", f"L_y{i}", f"L_z{i}"]

# -----------------------------
# DETECTAR FILAS INVÁLIDAS
# -----------------------------
invalid_rows = []

for idx, row in df.iterrows():

    right_hand = row[right_cols].values
    left_hand = row[left_cols].values

    # Mano derecha completamente en cero
    right_zero = np.all(right_hand == 0)

    # Mano izquierda completamente en cero
    left_zero = np.all(left_hand == 0)

    # Eliminar si cualquiera falta
    if right_zero or left_zero:
        invalid_rows.append(idx)

# -----------------------------
# ELIMINAR FILAS
# -----------------------------
df_clean = df.drop(index=invalid_rows)

print(f"Filas eliminadas: {len(invalid_rows)}")
print(f"Filas restantes: {len(df_clean)}")

# -----------------------------
# GUARDAR CSV LIMPIO
# -----------------------------
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\nCSV limpio guardado como:")
print(OUTPUT_FILE)