import pandas as pd

dataset = 'Toys_and_Games'
epoch = '26'

# 1. Carica i file TSV
df1 = pd.read_csv(f'{dataset}/edgelist_epoch_0.tsv', sep='\t')
df2 = pd.read_csv(f'{dataset}/edgelist_epoch_{epoch}.tsv', sep='\t')

# Assumiamo che le colonne si chiamino 'Source' e 'Target' come nel mio esempio precedente.
# Creiamo un "Set" di tuple (Sorgente, Destinazione) per ogni file.

# 2. Crea gli insiemi di archi
# zip crea coppie (src, tgt) molto velocemente
edges_1 = set(zip(df1['user'], df1['item']))
edges_2 = set(zip(df2['user'], df2['item']))

# 3. Calcolo Intersezione e Unione
intersection_count = len(edges_1.intersection(edges_2))
union_count = len(edges_1.union(edges_2))

# 4. Calcolo Jaccard
jaccard = intersection_count / union_count if union_count != 0 else 0

print(f"Archi in comune: {intersection_count}")
print(f"Totale archi unici: {union_count}")
print(f"Jaccard Similarity: {jaccard:.4f}")