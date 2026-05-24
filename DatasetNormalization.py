import pandas as pd
import numpy as np

# -----------------------------------
# CONFIG
# -----------------------------------
INPUT_FILE = "dirt_rally_multilabel_clean.csv"
OUTPUT_FILE = "dirt_rally_multilabel_normalized.csv"

# -----------------------------------
# CARGAR CSV
# -----------------------------------
df = pd.read_csv(INPUT_FILE)

print(f"\nFilas cargadas: {len(df)}")

# -----------------------------------
# FUNCION NORMALIZACION
# -----------------------------------
def normalize_hand(row, prefix):

    # Landmark 0 = wrist
    wrist_x = row[f"{prefix}_x0"]
    wrist_y = row[f"{prefix}_y0"]
    wrist_z = row[f"{prefix}_z0"]

    normalized = []

    for i in range(21):

        x = row[f"{prefix}_x{i}"] - wrist_x
        y = row[f"{prefix}_y{i}"] - wrist_y
        z = row[f"{prefix}_z{i}"] - wrist_z

        normalized.extend([x, y, z])

    return normalized

# -----------------------------------
# NORMALIZAR DATASET
# -----------------------------------
normalized_rows = []

for _, row in df.iterrows():

    # Mano derecha
    right_hand = normalize_hand(row, "R")

    # Mano izquierda
    left_hand = normalize_hand(row, "L")

    # Labels
    accelerate = row["accelerate"]
    brake = row["brake"]

    # Fila final
    final_row = right_hand + left_hand + [accelerate, brake]

    normalized_rows.append(final_row)

# -----------------------------------
# CREAR NUEVO DATAFRAME
# -----------------------------------
columns = []

# Columnas mano derecha
for i in range(21):
    columns += [f"R_x{i}", f"R_y{i}", f"R_z{i}"]

# Columnas mano izquierda
for i in range(21):
    columns += [f"L_x{i}", f"L_y{i}", f"L_z{i}"]

# Labels
columns += ["accelerate", "brake"]

df_normalized = pd.DataFrame(normalized_rows, columns=columns)

# -----------------------------------
# GUARDAR CSV
# -----------------------------------
df_normalized.to_csv(OUTPUT_FILE, index=False)

print("\nDataset normalizado guardado:")
print(OUTPUT_FILE)

print(f"\nTotal filas: {len(df_normalized)}")