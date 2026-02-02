import torch
import pandas as pd

dataset = 'Toys_and_Games'
epoch = '26'

# 1. Carica la matrice
matrix = torch.load(f'{dataset}/bi_adj_epoch_{epoch}.pt', map_location='cpu')

if matrix.is_sparse:
    matrix = matrix.to_dense()

# 2. Trova le coordinate dove esiste un collegamento (valore != 0)
# Questo estrae gli indici delle righe e delle colonne dove c'è un arco
rows, cols = torch.where(matrix != 0)
weights = matrix[rows, cols]

# 3. Crea un DataFrame con le connessioni
df_edges = pd.DataFrame({
    'user': rows.numpy(),
    'item': cols.numpy(),
    'weight': weights.numpy()
})

# 4. Salva in TSV
df_edges.to_csv(f'{dataset}/edgelist_epoch_{epoch}.tsv', sep='\t', index=False)

print(f"Salvato come lista di archi. Trovati {len(df_edges)} collegamenti.")