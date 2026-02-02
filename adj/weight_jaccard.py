import pandas as pd
import numpy as np

dataset = 'Toys_and_Games'
epoch = '26'

# 1. Carica i file
# Assumiamo che le righe siano nello stesso ordine.
# Se non sei sicuro dell'ordine, fammelo sapere e aggiungiamo un ordinamento preventivo.
df1 = pd.read_csv(f'{dataset}/edgelist_epoch_0.tsv', sep='\t')
df2 = pd.read_csv(f'{dataset}/edgelist_epoch_{epoch}.tsv', sep='\t')

# 2. Estrai solo i vettori dei pesi (convertendoli in numpy array per velocità)
w1 = df1['weight'].values
w2 = df2['weight'].values

# Verifica di sicurezza
if len(w1) != len(w2):
    raise ValueError("Attenzione: i file hanno un numero diverso di righe!")

# 3. Calcolo della Weighted Jaccard
# np.minimum confronta i due array elemento per elemento e prende il più piccolo
intersection = np.minimum(w1, w2).sum()
# np.maximum fa lo stesso prendendo il più grande
union = np.maximum(w1, w2).sum()

jaccard_weighted = intersection / union if union != 0 else 0

print(f"Weighted Jaccard Similarity: {jaccard_weighted:.4f}")

# EXTRA: Se ti interessa quanto sono distanti i valori mediamente
diff_media = np.mean(np.abs(w1 - w2))
print(f"Differenza media assoluta tra i pesi: {diff_media:.4f}")