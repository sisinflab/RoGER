import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.sparse as sp
import os

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_NAME = "Tools_and_Home"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, DATASET_NAME)
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, f"{DATASET_NAME}_png")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

N_USERS_DA_SELEZIONARE = 50
N_ITEMS_DA_SELEZIONARE = 50

pt_files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith(".pt")])

# --- 1. Trova i nodi una sola volta dal PRIMO file ---
first_pt_file = os.path.join(FOLDER_PATH, pt_files[0])
adj_tensor = torch.load(first_pt_file)
if adj_tensor.is_sparse:
    adj_tensor = adj_tensor.coalesce()
    values = adj_tensor.values().cpu().numpy()
    indices = adj_tensor.indices().cpu().numpy()
    shape = adj_tensor.shape
    bi_adj_matrix_sparse = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
else:
    bi_adj_matrix_sparse = sp.csr_matrix(adj_tensor.cpu().numpy())

n_users, n_items = bi_adj_matrix_sparse.shape
user_nodes = [f'u{i}' for i in range(n_users)]
item_nodes = [f'i{i}' for i in range(n_items)]

# --- SNOWBALL SAMPLING ---
G_full = nx.Graph()
G_full.add_nodes_from(user_nodes, bipartite=0)
G_full.add_nodes_from(item_nodes, bipartite=1)
bi_adj_coo = bi_adj_matrix_sparse.tocoo()
edges_full = [(f'u{u}', f'i{i}') for u, i in zip(bi_adj_coo.row, bi_adj_coo.col) if bi_adj_coo.data[(bi_adj_coo.row == u) & (bi_adj_coo.col == i)][0] > 0]
G_full.add_edges_from(edges_full)

# Scegli come seed l'utente più connesso
user_degrees = G_full.degree(user_nodes)
seed_user = max(user_degrees, key=lambda x: x[1])[0]

selected_users = set([seed_user])
selected_items = set()
frontier = set([seed_user])

while len(selected_users) < N_USERS_DA_SELEZIONARE or len(selected_items) < N_ITEMS_DA_SELEZIONARE:
    new_frontier = set()
    for node in frontier:
        neighbors = set(G_full.neighbors(node))
        if node.startswith('u'):
            # Espandi verso item
            for n in neighbors:
                if n not in selected_items and len(selected_items) < N_ITEMS_DA_SELEZIONARE:
                    selected_items.add(n)
                    new_frontier.add(n)
        else:
            # Espandi verso utenti
            for n in neighbors:
                if n not in selected_users and len(selected_users) < N_USERS_DA_SELEZIONARE:
                    selected_users.add(n)
                    new_frontier.add(n)
    if not new_frontier:
        break  # Non ci sono più nodi da espandere
    frontier = new_frontier

selected_nodes = list(selected_users) + list(selected_items)

# --- 2. Cicla su tutti i file, usa sempre gli stessi nodi ---
for pt_file in pt_files:
    MATRICE_FILE_PATH = os.path.join(FOLDER_PATH, pt_file)
    print(f"Caricamento matrice da {MATRICE_FILE_PATH}...")
    try:
        adj_tensor = torch.load(MATRICE_FILE_PATH)
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: {MATRICE_FILE_PATH}")
        continue

    if adj_tensor.is_sparse:
        adj_tensor = adj_tensor.coalesce()
        values = adj_tensor.values().cpu().numpy()
        indices = adj_tensor.indices().cpu().numpy()
        shape = adj_tensor.shape
        bi_adj_matrix_sparse = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    else:
        bi_adj_matrix_sparse = sp.csr_matrix(adj_tensor.cpu().numpy())

    user_nodes = [f'u{i}' for i in range(bi_adj_matrix_sparse.shape[0])]
    item_nodes = [f'i{i}' for i in range(bi_adj_matrix_sparse.shape[1])]

    G = nx.Graph()
    G.add_nodes_from(user_nodes, bipartite=0)
    G.add_nodes_from(item_nodes, bipartite=1)

    bi_adj_coo = bi_adj_matrix_sparse.tocoo()
    edges = []
    alphas = []
    for u_idx, i_idx, rating in zip(bi_adj_coo.row, bi_adj_coo.col, bi_adj_coo.data):
        if rating > 0:
            u = f'u{u_idx}'
            i = f'i{i_idx}'
            edges.append((u, i))
            alphas.append(np.clip(rating, 0.0, 1.0))

    G.add_edges_from(edges)

    G_sub = G.subgraph(selected_nodes)

    user_nodes_sub = {n for n, d in G_sub.nodes(data=True) if d['bipartite'] == 0}
    item_nodes_sub = {n for n, d in G_sub.nodes(data=True) if d['bipartite'] == 1}

    if not user_nodes_sub or not item_nodes_sub:
        print(f"IMPOSSIBILE DISEGNARE {pt_file}: sottografo vuoto o senza utenti/item.")
        continue

    pos = nx.bipartite_layout(G_sub, user_nodes_sub)
    plt.figure(figsize=(18, 12))
    nx.draw_networkx_nodes(G_sub, pos, nodelist=user_nodes_sub, node_color='dodgerblue', label='Utenti', node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(G_sub, pos, nodelist=item_nodes_sub, node_color='springgreen', label='Item', node_size=300, alpha=0.8)

    # Disegna archi con alpha proporzionale al rating
    for (edge, alpha) in zip(edges, alphas):
        if edge[0] in selected_nodes and edge[1] in selected_nodes:
            nx.draw_networkx_edges(G_sub, pos, edgelist=[edge], edge_color='gray', alpha=alpha)

    plt.title(f"{pt_file} - Sottografo Bipartito (Sample {N_USERS_DA_SELEZIONARE}+{N_ITEMS_DA_SELEZIONARE})", fontsize=16)
    plt.legend(scatterpoints=1, loc='best')
    plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, pt_file.replace('.pt', '.png'))
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Salvato: {out_path}")