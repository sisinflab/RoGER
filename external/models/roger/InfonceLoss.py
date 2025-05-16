import torch
import torch.nn.functional as F

def infonce_loss(z1, z2, temperature=0.5):
    # z1, z2: [batch_size, embed_dim]
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)  # [2*B, D]

    # Similarity matrix
    sim_matrix = torch.matmul(representations, representations.T) / temperature

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Positive pairs: i-th in z1 with i-th in z2 (and viceversa)
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss