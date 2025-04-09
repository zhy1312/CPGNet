from dgl.data.utils import load_graphs, save_graphs
import os
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch
import gc

dataset_name = "YNLUAD"
graph_path = r"graphs/YNLUAD/feature"
save_path = "graphs/YNLUAD/feature"


if not os.path.exists(save_path):
    os.makedirs(save_path)


def random_walk_pe(g, k):
    A = g.adj_external(scipy_fmt="csr")  # adjacency matrix
    RW = torch.tensor(A / (A.sum(1) + 1e-30)).cuda()  # 1-step transition probability
    # Iterate for k steps
    PE = [RW.diagonal()]
    RW_power = RW
    for _ in range(k - 1):
        RW_power = RW_power @ RW
        PE.append(RW_power.diagonal())
    RPE = torch.stack(PE, dim=-1).float()
    del A, RW, PE, RW_power
    torch.cuda.empty_cache()
    gc.collect()
    return RPE.cpu()


def random_walk_pe_cpu(g, k):
    A = g.adj_external(scipy_fmt="csr")  # adjacency matrix
    RW = torch.tensor(A / (A.sum(1) + 1e-30))  # 1-step transition probability
    # Iterate for k steps
    PE = [RW.diagonal()]
    RW_power = RW
    for _ in range(k - 1):
        RW_power = RW_power @ RW
        PE.append(RW_power.diagonal())
    RPE = torch.stack(PE, dim=-1).float()
    del A, RW, PE, RW_power
    torch.cuda.empty_cache()
    gc.collect()
    return RPE


graph_list = os.listdir(graph_path)
for graph_name in tqdm(graph_list):
    graph_path = os.path.join(graph_path, graph_name)
    graph = load_graphs(graph_path)
    g = graph[0]
    start, end = g.edges()
    node_type = torch.zeros(g.num_nodes(), dtype=torch.uint8)
    feat = g.ndata["feat"]
    centroid = g.ndata["centroid"]
    # Create edge types
    edge_sim = []
    edge_Dist = []
    for idx_a, idx_b in zip(start, end):
        corr = 1 - cosine(feat[idx_a], feat[idx_b])
        a = centroid[idx_a]
        b = centroid[idx_b]
        edge_sim.append([torch.tensor(corr, dtype=torch.float)])
        edge_Dist.append([torch.sqrt(torch.pow(a - b, 2).sum())])
    edge_sim = torch.tensor(edge_sim)
    edge_Dist = torch.tensor(edge_Dist)
    efeat = torch.cat([edge_sim, edge_Dist], dim=1)
    mean = torch.mean(efeat, axis=0)
    std = torch.std(efeat, axis=0)
    norm_efeat = (efeat - mean) / std
    g.edata.update({"feat": norm_efeat})
    try:
        g.ndata["PE"] = random_walk_pe(g, k=20)
    except:
        g.ndata["PE"] = random_walk_pe_cpu(g, k=20)
    save_graphs(os.path.join(save_path, graph_name), g)
