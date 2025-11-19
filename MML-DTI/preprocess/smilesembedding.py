import os
import torch
import torch as th
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch.optim.optimizer import Optimizer, required
from transformers import AutoTokenizer, AutoModel
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import torch.nn.functional as F
from rdkit.Chem import AllChem
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import gc
import warnings
import geoopt



os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"
warnings.filterwarnings('ignore', category=UserWarning)


def th_dot(x, y, keepdim=False):
    return (x * y).sum(dim=-1, keepdim=keepdim)


def th_norm(x, eps=1e-5):
    return torch.sqrt(th_dot(x, x, keepdim=True) + eps)


def clip_by_norm(x, max_norm):
    norm = th_norm(x)
    mask = (norm > max_norm).float()
    return mask * (x / norm) * max_norm + (1 - mask) * x


def th_atanh(x, eps=1e-5):
    return 0.5 * torch.log((1 + x.clamp(-1 + eps, 1 - eps)) / (1 - x.clamp(-1 + eps, 1 - eps) + eps))


# Poincare distance
class PoincareDistance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx).clamp(min=eps)
        beta = (1 - sqnormv).clamp(min=eps)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2)) \
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1 + eps)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm) + eps) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1 + eps)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None


# Custom collate_fn to handle graph data of varying sizes
def graph_collate_fn(batch):
    collated = {
        'node_features': [],
        'adj_mat': [],
        'weight': [],
        'mask': [],
        'num_atoms': [],
        'original_indices': []  # Store original index information
    }

    for i, graph in enumerate(batch):
        collated['node_features'].append(graph['node_features'])
        collated['adj_mat'].append(graph['adj_mat'])
        collated['weight'].append(graph['weight'])
        collated['mask'].append(graph['mask'])
        collated['num_atoms'].append(graph['num_atoms'])
        collated['original_indices'].append(i)  # Save original index

    return collated


class PoincareManifold:
    def __init__(self, c=1.0, EPS=1e-5, PROJ_EPS=1e-5):
        self.c = c  # Curvature
        self.EPS = EPS
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()

    def normalize(self, x):
        return clip_by_norm(x, (1. - self.PROJ_EPS))

    def init_embed(self, embed, irange=1e-2):
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def mob_add(self, u, v):
        """Add two vectors in hyperbolic space"""
        v = v + self.EPS

        # Ensure u and v have the same shape
        if u.shape != v.shape:
            min_dim = min(u.dim(), v.dim())
            for i in range(min_dim):
                if u.size(i) != v.size(i):
                    v = v.expand_as(u)

        th_dot_u_v = 2. * th_dot(u, v, keepdim=True)
        th_norm_u_sq = th_dot(u, u, keepdim=True)
        th_norm_v_sq = th_dot(v, v, keepdim=True)
        denominator = 1. + th_dot_u_v + th_norm_v_sq * th_norm_u_sq
        result = (1. + th_dot_u_v + th_norm_v_sq) / (denominator + self.EPS) * u + \
                 (1. - th_norm_u_sq) / (denominator + self.EPS) * v
        return self.normalize(result)

    def distance(self, u, v):
        return PoincareDistance.apply(u, v, self.EPS)

    def lambda_x(self, x):
        """Conformal factor"""
        return 2. / (1 - th_dot(x, x, keepdim=True) + self.EPS)

    def log_map_zero(self, y):
        diff = y + self.EPS
        norm_diff = th_norm(diff)
        return 1. / th_atanh(norm_diff, self.EPS) / (norm_diff + self.EPS) * diff

    def log_map_x(self, x, y):
        diff = self.mob_add(-x, y) + self.EPS
        norm_diff = th_norm(diff)
        lam = self.lambda_x(x)
        return ((2. / lam) * th_atanh(norm_diff, self.EPS) / (norm_diff + self.EPS)) * diff

    def metric_tensor(self, x, u, v):
        """Compute the metric tensor in hyperbolic space."""
        if x.shape != u.shape or u.shape != v.shape:
            u = u.expand_as(x)
            v = v.expand_as(x)

        u_dot_v = th_dot(u, v, keepdim=True)
        lambda_x = self.lambda_x(x)
        return lambda_x * lambda_x * u_dot_v

    def exp_map_zero(self, v):
        """Exponential map from tangent space at zero to hyperbolic space"""
        v = v + self.EPS
        norm_v = th_norm(v)
        # Avoid division by zero
        safe_norm_v = torch.where(norm_v < self.EPS, torch.ones_like(norm_v) * self.EPS, norm_v)
        result = self.tanh(safe_norm_v) / (safe_norm_v + self.EPS) * v
        return self.normalize(result)

    def exp_map_x(self, x, v):
        """Exponential map from tangent space at x to hyperbolic space"""
        v = v + self.EPS  # Perturb v to avoid dealing with v = 0

        # Ensure x and v have compatible shapes
        if x.shape != v.shape:
            # Calculate required dimensions for broadcasting
            new_shape = []
            for i in range(max(x.dim(), v.dim())):
                x_dim = x.size(i) if i < x.dim() else 1
                v_dim = v.size(i) if i < v.dim() else 1

                if x_dim != v_dim:
                    if x_dim == 1:
                        x = x.expand_as(v)
                    elif v_dim == 1:
                        v = v.expand_as(x)
                    else:
                        # Attempt to broadcast
                        try:
                            x, v = torch.broadcast_tensors(x, v)
                        except:
                            # If broadcasting fails, create new tensors
                            new_shape.append(max(x_dim, v_dim))
                            if i < x.dim():
                                x = x.unsqueeze(-1).expand(*new_shape)
                            if i < v.dim():
                                v = v.unsqueeze(-1).expand(*new_shape)

        norm_v = th_norm(v)
        safe_norm_v = torch.where(norm_v < self.EPS, torch.ones_like(norm_v) * self.EPS, norm_v)
        lam = self.lambda_x(x)

        # Ensure lam and safe_norm_v have compatible dimensions
        if lam.dim() < safe_norm_v.dim():
            lam = lam.unsqueeze(-1).expand_as(safe_norm_v)
        elif safe_norm_v.dim() < lam.dim():
            safe_norm_v = safe_norm_v.unsqueeze(-1).expand_as(lam)

        # Calculate the second term with dimension matching
        scale = self.tanh(lam * safe_norm_v / 2) / (safe_norm_v + self.EPS)
        second_term = scale * v

        return self.normalize(self.mob_add(x, second_term))

    def gyr(self, u, v, w):
        # Ensure all input tensors have the same shape
        if u.shape != v.shape or v.shape != w.shape:
            w = w.expand_as(u)

        u_norm = th_dot(u, u, keepdim=True)
        v_norm = th_dot(v, v, keepdim=True)
        u_dot_w = th_dot(u, w, keepdim=True)
        v_dot_w = th_dot(v, w, keepdim=True)
        u_dot_v = th_dot(u, v, keepdim=True)

        A = - u_dot_w * v_norm + v_dot_w + 2 * u_dot_v * v_dot_w
        B = - v_dot_w * u_norm - u_dot_w
        D = 1 + 2 * u_dot_v + u_norm * v_norm
        return w + 2 * (A * u + B * v) / (D + self.EPS)

    def parallel_transport(self, src, dst, v):
        # Ensure all input tensors have the same shape
        if src.shape != dst.shape or dst.shape != v.shape:
            v = v.expand_as(src)

        lambda_src = self.lambda_x(src)
        lambda_dst = th.clamp(self.lambda_x(dst), min=self.EPS)
        return lambda_src / lambda_dst * self.gyr(dst, -src, v)

    def rgrad(self, p, d_p):
        """Convert Euclidean gradient to Riemannian gradient in the Poincare ball."""
        p_sqnorm = th_dot(p, p, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4.0)
        return d_p

    def dist(self, u, v):
        """Compute hyperbolic distance"""
        return self.distance(u, v)

    def proj(self, x):
        """Project points back into the hyperbolic ball"""
        norm = th_norm(x)
        mask = (norm >= 1.0).float()
        safe_norm = torch.where(norm < self.EPS, torch.ones_like(norm) * self.EPS, norm)
        return (1 - mask) * x + mask * (x / safe_norm * (1 - self.PROJ_EPS))


class CentroidDistance(nn.Module):
    def __init__(self, args, logger, manifold, embed_size, num_centroids):
        super().__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.num_centroids = num_centroids
        self.embed_size = embed_size

        # Centroid embedding
        self.centroid_embedding = nn.Embedding(
            num_centroids, embed_size,
            sparse=False,
            scale_grad_by_freq=False,
        )
        if args.embed_manifold == 'hyperbolic':
            # Use our own init_embed method
            self.manifold.init_embed(self.centroid_embedding)
        elif args.embed_manifold == 'euclidean':
            nn.init.xavier_uniform_(self.centroid_embedding.weight)

    def forward(self, node_repr, mask):
        """
        Args:
            node_repr: [node_num, embed_size] Node representations in hyperbolic space
            mask: [node_num, 1] 1 for real nodes, 0 for padding nodes
        return:
            graph_centroid_dist: [1, num_centroids]
            node_centroid_dist: [1, node_num, num_centroids]
        """
        node_num = node_repr.size(0)

        # Broadcast and reshape node_repr to [node_num * num_centroids, embed_size]
        node_repr = node_repr.unsqueeze(1).expand(
            -1, self.num_centroids, -1
        ).contiguous().view(-1, self.embed_size)

        # Broadcast and reshape centroid embeddings to [node_num * num_centroids, embed_size]
        centroid_repr = self.centroid_embedding(
            torch.arange(self.num_centroids, device=node_repr.device)
        )
        centroid_repr = centroid_repr.unsqueeze(0).expand(
            node_num, -1, -1
        ).contiguous().view(-1, self.embed_size)

        # Compute distance in hyperbolic space
        node_centroid_dist = self.manifold.distance(node_repr, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.num_centroids) * mask

        # Average pooling over nodes
        graph_centroid_dist = torch.sum(node_centroid_dist, dim=1) / torch.sum(mask)
        return graph_centroid_dist, node_centroid_dist


class RiemannianAMSGrad(Optimizer):
    def __init__(self, params, lr, manifold, betas=(0.9, 0.99), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.manifold = manifold

    def step(self, lr=None):
        """Perform a single optimization step"""
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    grad = self.manifold.rgrad(p, grad)

                    if lr is None:
                        lr = group['lr']

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['tau'] = torch.zeros_like(p.data)
                        # Exponential moving average of gradient
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        # Maintains max of all exp. moving avg. of sq. grad.
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq, tau, max_exp_avg_sq = (
                        state['exp_avg'], state['exp_avg_sq'],
                        state['tau'], state['max_exp_avg_sq']
                    )

                    beta1, beta2 = group['betas']
                    state['step'] += 1

                    # Update first and second moment
                    exp_avg.data = beta1 * tau + (1 - beta1) * grad

                    # Compute metric tensor
                    metric = self.manifold.metric_tensor(p, grad, grad)
                    # Ensure metric has the same shape as exp_avg_sq
                    if metric.shape != exp_avg_sq.shape:
                        metric = metric.expand_as(exp_avg_sq)

                    # Fix deprecated warning: use new add_ signature
                    exp_avg_sq.mul_(beta2).add_(metric, alpha=1 - beta2)
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use max for gradient normalization
                    denom = max_exp_avg_sq.sqrt().clamp_(min=group['eps'])
                    step_size = group['lr']

                    p_original = p.clone()
                    # Ensure denominator matches exp_avg shape
                    if denom.shape != exp_avg.shape:
                        denom = denom.expand_as(exp_avg)

                    before_proj = self.manifold.exp_map_x(p, -step_size * exp_avg / denom)
                    p.data = self.manifold.proj(before_proj)
                    tau.data = self.manifold.parallel_transport(p_original, p, exp_avg)
        return loss


# Hyperbolic Graph Neural Network implementation
class RiemannianGNN(nn.Module):
    def __init__(self, args, manifold, input_dim, num_centroids=10):
        super().__init__()
        self.args = args
        self.manifold = manifold
        self.input_dim = input_dim
        self.num_centroids = num_centroids

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, args.embed_size)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # Output projection layer
        self.output_proj = nn.Linear(args.embed_size, input_dim)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # Set up graph neural network parameters
        self.set_up_params()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # 添加残差投影层
        self.residual_proj = nn.Linear(input_dim, args.embed_size)
        nn.init.xavier_uniform_(self.residual_proj.weight)

        # Centroid distance calculation module
        self.centroid_distance = CentroidDistance(
            args, None, manifold, args.embed_size, num_centroids
        )

    def create_params(self):
        msg_weight = []
        for _ in range(self.args.gnn_layer):
            M = nn.Parameter(torch.zeros([self.args.embed_size, self.args.embed_size]))
            nn.init.xavier_uniform_(M)
            msg_weight.append(M)
        return nn.ParameterList(msg_weight)

    def set_up_params(self):
        self.type_of_msg = 1
        for i in range(0, self.type_of_msg):
            setattr(self, "msg_%d_weight" % i, self.create_params())

    def retrieve_params(self, weight, step):
        return weight[step]

    def apply_activation(self, node_repr):
        """Apply non-linear activation function in hyperbolic space"""
        poincare_repr = self.manifold.log_map_zero(node_repr)
        activated = self.activation(poincare_repr)
        return self.manifold.exp_map_zero(activated)

    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask):
        """Aggregate neighbor information"""
        transformed = torch.mm(
            self.manifold.log_map_zero(node_repr),
            layer_weight
        ) * mask
        combined_msg = torch.mm(adj_mat * weight, transformed)
        return self.manifold.exp_map_zero(combined_msg)

    def get_combined_msg(self, step, node_repr, adj_mat, weight, mask):
        layer_weight = self.retrieve_params(getattr(self, "msg_0_weight"), step)
        return self.aggregate_msg(
            node_repr, adj_mat, weight, layer_weight, mask
        )

    def forward(self, node_repr, adj_mat, weight, mask):
        # Save original features for reconstruction loss
        original_features = node_repr

        # Project input features to hyperbolic embedding space
        node_repr = self.manifold.exp_map_zero(self.input_proj(node_repr))

        # 计算残差投影
        residual_projection = self.manifold.exp_map_zero(
            self.residual_proj(original_features)
        )

        # Graph neural network layers with residual connections
        for step in range(self.args.gnn_layer):
            # 保存当前节点表示作为残差
            residual = node_repr

            # Message passing in hyperbolic space
            combined_msg = self.get_combined_msg(
                step, node_repr, adj_mat, weight, mask
            )
            combined_msg = self.dropout(combined_msg) * mask

            # Update node representation
            node_repr = combined_msg
            node_repr = self.apply_activation(node_repr) * mask

            # 添加残差连接 - 使用莫比乌斯加法
            if step == 0:  # 第一层使用原始残差投影
                node_repr = self.manifold.mob_add(residual_projection, node_repr)
            else:  # 后续层使用前一层输出作为残差
                node_repr = self.manifold.mob_add(residual, node_repr)

            # 确保结果在双曲球内
            node_repr = self.manifold.proj(node_repr)

        # Compute graph-level centroid distance
        graph_centroid_dist, _ = self.centroid_distance(node_repr, mask)

        # Reconstruct features
        euclidean_repr = self.manifold.log_map_zero(node_repr)
        reconstructed_features = self.output_proj(euclidean_repr)

        return node_repr, None, graph_centroid_dist.squeeze(0), reconstructed_features, original_features

# Hyperbolic projection layer
class HyperbolicProjection(nn.Module):
    def __init__(self, input_dim=256, output_dim=128, manifold=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.manifold = manifold or PoincareManifold()

    def forward(self, x):
        euclidean_proj = self.activation(self.linear(x))
        return self.manifold.exp_map_zero(euclidean_proj)


class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# Hyperbolic GNN training
def train_hgnn(graphs, device, args):
    # Create dataset
    dataset = GraphDataset(graphs)

    # Use custom collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=graph_collate_fn
    )

    # Determine input dimension
    input_dim = graphs[0]['node_features'].shape[1]

    # Initialize manifold and model
    manifold = PoincareManifold(c=args.curvature)
    model = RiemannianGNN(args, manifold, input_dim).to(device)

    # Use Riemannian AMSGrad optimizer
    optimizer = RiemannianAMSGrad(model.parameters(), lr=args.lr, manifold=manifold)

    # Initialize early stopping related variables
    best_loss = float('inf')
    best_model_state = None
    best_manifold = None
    best_features = None
    epochs_no_improve = 0
    early_stop_patience = 20
    min_delta = 0.001  # Minimum improvement threshold

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        processed_batches = 0
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(batch_iter):
            optimizer.zero_grad()
            batch_loss = 0
            processed_graphs = 0

            # Process each graph in the batch
            for i in range(len(batch['node_features'])):
                try:
                    node_features = batch['node_features'][i].to(device)
                    adj_mat = batch['adj_mat'][i].to(device)
                    weight = batch['weight'][i].to(device)
                    mask = batch['mask'][i].to(device)
                    num_atoms = batch['num_atoms'][i]

                    # Forward propagation
                    _, _, _, reconstructed, original = model(
                        node_features, adj_mat, weight, mask
                    )

                    # Check for NaN values
                    if torch.isnan(reconstructed).any() or torch.isnan(original).any():
                        continue

                    # Loss function - feature reconstruction
                    loss = F.mse_loss(reconstructed, original)
                    batch_loss += loss.item()
                    processed_graphs += 1

                    # Backward propagation
                    loss.backward()

                    # Manually release memory
                    del node_features, adj_mat, weight, mask, reconstructed, original, loss
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f" Graph processing error (index {batch['original_indices'][i]}): {str(e)}")
                    continue

            if processed_graphs == 0:
                batch_iter.set_postfix(loss="skipped")
                continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer update
            optimizer.step()

            # Update loss statistics
            avg_batch_loss = batch_loss / processed_graphs if processed_graphs > 0 else 0
            total_loss += avg_batch_loss
            processed_batches += 1

            # Update progress bar
            batch_iter.set_postfix(loss=avg_batch_loss)

            # Clean memory
            del batch
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate average epoch loss
        if processed_batches > 0:
            avg_epoch_loss = total_loss / processed_batches
        else:
            avg_epoch_loss = total_loss

        print(f"Epoch {epoch + 1}, Avg Loss: {avg_epoch_loss:.4f}")

        # Early stopping mechanism check
        if avg_epoch_loss < best_loss - min_delta:
            # Save best model and features
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict()
            best_manifold = manifold

            # Generate and save current best features
            with torch.no_grad():
                best_features = []
                for graph in tqdm(graphs, desc="Generating best embeddings"):

                    node_features = graph['node_features'].to(device)
                    adj_mat = graph['adj_mat'].to(device)
                    weight = graph['weight'].to(device)
                    mask = graph['mask'].to(device)

                    _, _, graph_embedding, _, _ = model(
                        node_features, adj_mat, weight, mask
                    )
                    # Ensure graph embedding is 2D
                    best_features.append(graph_embedding.cpu().numpy())

            best_features = np.array(best_features)
            epochs_no_improve = 0
            print(f"Updated best model, loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve}/{early_stop_patience}")

        # Check if early stopping condition is met
        if epochs_no_improve >= early_stop_patience:
            print(f" Early stopping triggered! {early_stop_patience} consecutive epochs without significant improvement")
            break

        # Clean memory
        torch.cuda.empty_cache()
        gc.collect()

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (loss: {best_loss:.4f})")
    else:
        print(" No best model state found, using final model")

    return model, manifold, best_features


# Morgan fingerprint generation
def generate_morgan_fingerprints(smiles_list, radius=2, n_bits=256):
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Generating Morgan fingerprints"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fp_array = np.array(fp, dtype=np.float32)
            fingerprints.append(fp_array)
        except:
            fingerprints.append(np.random.rand(n_bits))
    return np.array(fingerprints)


# Dataset information
def dataset_info(dataset):
    atom_types = [
        'Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
        'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)',
        'S1(-1)', 'P5(0)', 'B3(0)'
    ]
    metal_ions = ['Mg', 'Zn', 'Fe', 'Na', 'K', 'Ca', 'Li', 'Al', 'Cu', 'Mn']
    for metal in metal_ions:
        atom_types.append(f"{metal}0(0)")
        atom_types.append(f"{metal}0(1)")
        atom_types.append(f"{metal}0(2)")
        atom_types.append(f"{metal}0(-1)")

    return {
        'atom_types': atom_types,
        'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O',
                           11: 'S', 12: 'S', 13: 'S', 14: 'S', 15: 'P', 16: 'B'},
    }


# Helper functions
def onehot(idx, length):
    return [1 if i == idx else 0 for i in range(length)]


def get_atom_type(atom, atom_types):
    symbol = atom.GetSymbol()
    valence = atom.GetTotalValence()
    charge = atom.GetFormalCharge()
    atom_str = f"{symbol}{valence}({charge})"

    if atom_str in atom_types:
        return atom_types.index(atom_str)

    # Metal ion handling
    if symbol in ['Mg', 'Zn', 'Fe', 'Na', 'K', 'Ca', 'Li', 'Al', 'Cu', 'Mn']:
        virtual_type = f"{symbol}0({charge})"
        if virtual_type in atom_types:
            return atom_types.index(virtual_type)
        neutral_type = f"{symbol}0(0)"
        if neutral_type in atom_types:
            return atom_types.index(neutral_type)

    # Default mapping
    defaults = {
        'C': [('C4(0)', 1.0)], 'N': [('N3(0)', 0.7), ('N4(1)', 0.3)],
        'O': [('O2(0)', 0.8), ('O1(-1)', 0.2)], 'S': [('S2(0)', 0.6), ('S4(0)', 0.3), ('S6(0)', 0.1)],
        'F': [('F1(0)', 1.0)], 'Cl': [('Cl1(0)', 1.0)], 'Br': [('Br1(0)', 1.0)],
        'I': [('I1(0)', 1.0)], 'H': [('H1(0)', 1.0)], 'P': [('P5(0)', 1.0)], 'B': [('B3(0)', 1.0)],
    }

    if symbol in defaults:
        for type_str, prob in defaults[symbol]:
            if type_str in atom_types and random.random() < prob:
                return atom_types.index(type_str)

    similar_types = [t for t in atom_types if t.startswith(symbol)]
    if similar_types:
        return atom_types.index(similar_types[0])

    return 1


def to_graph(smiles, dataset, output_dir=None, index=0):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                                      Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except:
                    Chem.Kekulize(mol, clearAromaticFlags=True)

        if mol is None:
            print(f" Unable to parse SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)
        atom_types = dataset_info(dataset)['atom_types']

        # Node features
        nodes = []
        for atom in mol.GetAtoms():
            try:
                atom_idx = get_atom_type(atom, atom_types)
                nodes.append(onehot(atom_idx, len(atom_types)))
            except:
                # Default to carbon atom features
                nodes.append(onehot(1, len(atom_types)))

        # Edge processing
        num_atoms = mol.GetNumAtoms()
        adj_mat = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        weight_mat = np.zeros((num_atoms, num_atoms), dtype=np.float32)

        for bond in mol.GetBonds():
            try:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # Bond type handling
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    bond_code = 2.0
                elif bond.GetBondType() == Chem.BondType.TRIPLE:
                    bond_code = 3.0
                elif bond.GetBondType() == Chem.BondType.AROMATIC:
                    bond_code = 1.5  # Special handling for aromatic bonds
                else:
                    bond_code = 1.0

                adj_mat[i, j] = 1.0
                adj_mat[j, i] = 1.0
                weight_mat[i, j] = bond_code
                weight_mat[j, i] = bond_code
            except:
                continue

        # Add self-loops
        for i in range(num_atoms):
            adj_mat[i, i] = 1.0
            weight_mat[i, i] = 1.0

        # Convert to PyTorch tensors
        node_features = torch.tensor(nodes, dtype=torch.float32)
        adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
        weight = torch.tensor(weight_mat, dtype=torch.float32)
        mask = torch.ones(num_atoms, 1, dtype=torch.float32)

        graph_data = {
            'node_features': node_features,
            'adj_mat': adj_mat,
            'weight': weight,
            'mask': mask,
            'num_atoms': num_atoms
        }

        return graph_data

    except Exception as e:
        print(f"Molecule processing error: {smiles} - {str(e)}")
        return None


# Hyperbolic space fusion function
def hyperbolic_fusion(graph_emb, morgan_emb, curvature=1.0, device="cpu"):
    manifold = geoopt.PoincareBall(c=curvature)
    projector = HyperbolicProjection().to(device)
    morgan_projected = projector(torch.tensor(morgan_emb).float().to(device))

    graph_point = manifold.expmap0(torch.tensor(graph_emb).float().to(device))
    morgan_point = manifold.expmap0(morgan_projected)

    fused_point = manifold.mobius_add(graph_point, morgan_point)
    return fused_point.detach().cpu().numpy()


# SMILES validation
def validate_smiles(smiles_list):
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_smiles.append(smiles)
            else:
                print(f" Invalid SMILES: {smiles}")
        except:
            pass

    print(f" Valid SMILES count: {len(valid_smiles)}/{len(smiles_list)}")
    return valid_smiles


# Main processing function (with enhanced memory management)
def process_smiles_to_hyperbolic(smiles_list, dataset, output_dir, device, args):
    all_graphs = []
    valid_smiles = []

    # Build graph data
    for i, smiles in enumerate(tqdm(smiles_list, desc="Building molecular graphs")):
        graph_data = to_graph(smiles, dataset, output_dir, i)
        if graph_data is not None:
            all_graphs.append(graph_data)
            valid_smiles.append(smiles)

            # Periodically clean memory
            if len(all_graphs) % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(f" Successfully built {len(all_graphs)} molecular graphs")

    if len(all_graphs) == 0:
        print(" No valid molecular graphs, exiting")
        return None, None, valid_smiles

    # Train hyperbolic GNN
    print(" Starting hyperbolic graph neural network training...")
    hgnn_model, manifold, all_hyperbolic_features = train_hgnn(all_graphs, device, args)

    if hgnn_model is None or manifold is None:
        print(" Hyperbolic graph neural network training failed")
        return None, None, valid_smiles

    # Generate graph embedding features
    all_hyperbolic_features = []
    for graph in tqdm(all_graphs, desc="Generating graph embeddings"):
        node_features = graph['node_features'].to(device)
        adj_mat = graph['adj_mat'].to(device)
        weight = graph['weight'].to(device)
        mask = graph['mask'].to(device)

        with torch.no_grad():
            # Forward propagation to get graph embedding
            _, _, graph_embedding, _, _ = hgnn_model(node_features, adj_mat, weight, mask)
            # Ensure graph embedding is 2D
            all_hyperbolic_features.append(graph_embedding.cpu().numpy())

        # Clean intermediate variables
        del node_features, adj_mat, weight, mask
        torch.cuda.empty_cache()

    all_hyperbolic_features = np.array(all_hyperbolic_features)

    return all_hyperbolic_features, valid_smiles


# Generate pretrained model embeddings (with error handling)
def generate_molformer_embeddings(smiles_list, device, max_length=512):
    try:
        tokenizer = AutoTokenizer.from_pretrained("./molformer", trust_remote_code=True)
        model = AutoModel.from_pretrained("./molformer", trust_remote_code=True)
        hidden_size = model.config.hidden_size
        print(f"Using local pretrained model, hidden_size: {hidden_size}")

        model.to(device)
        model.eval()

        embeddings = []
        for smiles in tqdm(smiles_list, desc="Generating pretrained model embeddings"):
            try:
                inputs = tokenizer(
                    smiles,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    seq_embedding = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(seq_embedding.cpu().numpy().flatten())

                # Clean memory
                del inputs, outputs
                torch.cuda.empty_cache()

            except Exception as e:
                print(f" Embedding generation error for SMILES: {smiles[:30]}... - {str(e)}")
                embeddings.append(np.zeros(hidden_size))

        return np.array(embeddings)
    except Exception as e:
        print(f"Failed to load Molformer model: {str(e)}")
        return np.random.rand(len(smiles_list), 768)  # Return random embeddings as fallback


# Hyperbolic space fusion model
class ConcatFusion(nn.Module):
    def __init__(self, seq_dim, hyp_fused_dim, hidden_dim=256):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(seq_dim + hyp_fused_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, seq_emb, hyp_fused_emb):
        # Ensure input dimensions are correct
        assert seq_emb.dim() == 2, f"seq_emb should be 2D, got {seq_emb.dim()}"
        assert hyp_fused_emb.dim() == 2, f"hyp_fused_emb should be 2D, got {hyp_fused_emb.dim()}"

        combined = torch.cat([seq_emb, hyp_fused_emb], dim=1)
        #Do not use linear layer, we will comment it out later
        fused = self.fusion_layer(combined)
        return fused

def set_seed(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main function (optimized memory management)
def get_smiles_embeddings(dir_path, dataset_name,save_path,device):


    set_seed()  # Call to fix random seed
    # Parameter configuration
    class Args:
        def __init__(self):
            self.embed_size = 128
            self.gnn_layer = 2
            self.dropout = 0.1
            self.lr = 0.001
            self.epochs = 300
            self.curvature = 1.0
            self.embed_manifold = 'hyperbolic'
            self.batch_size = 64

    args = Args()

    # Read CSV file
    csv_path = os.path.join(dir_path, "smiles.csv")
    if not os.path.exists(csv_path):
        print(f"File does not exist: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f" Successfully read CSV file, total records: {len(df)}")
    smiles_list = df.smiles.values.tolist()
    smiles_list = validate_smiles(smiles_list)

    if not smiles_list:
        print(" No valid SMILES data")
        return

    # Create output directory
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Generate graph structure features (using HGNN)
    graph_features, valid_smiles = process_smiles_to_hyperbolic(
        smiles_list, dataset_name, dir_path, device, args
    )

    if graph_features is None:
        print(" Unable to generate graph features")
        return

    # Step 2: Generate Morgan fingerprint features
    morgan_features = generate_morgan_fingerprints(valid_smiles)
    # np.save(os.path.join(dir_path, 'morgan_features.npy'), morgan_features)
    print(f"Morgan fingerprint shape: {morgan_features.shape}")

    # Step 3: Hyperbolic space fusion
    fused_hyperbolic_features = []
    for i in tqdm(range(len(graph_features)), desc="Hyperbolic space fusion"):
        try:
            fused_feat = hyperbolic_fusion(
                graph_features[i],
                morgan_features[i],
                curvature=args.curvature,
                device=device
            )
            fused_hyperbolic_features.append(fused_feat)
        except:
            fused_hyperbolic_features.append(np.zeros_like(graph_features[i]))

        # Periodically clean memory
        if i % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    fused_hyperbolic_features = np.array(fused_hyperbolic_features)
    # np.save(os.path.join(dir_path, 'fused_hyperbolic_features.npy'), fused_hyperbolic_features)

    # Step 4: Generate sequence embeddings
    seq_embeddings = generate_molformer_embeddings(valid_smiles, device)

    # Ensure dimensions match
    min_len = min(len(fused_hyperbolic_features), len(seq_embeddings))
    fused_hyperbolic_features = fused_hyperbolic_features[:min_len]
    seq_embeddings = seq_embeddings[:min_len]

    # Convert to PyTorch tensors
    hyp_fused_tensor = torch.tensor(fused_hyperbolic_features, dtype=torch.float32).to(device)
    seq_tensor = torch.tensor(seq_embeddings, dtype=torch.float32).to(device)

    # Initialize fusion model
    fusion_model = ConcatFusion(
        seq_dim=seq_tensor.size(1),
        hyp_fused_dim=hyp_fused_tensor.size(1)
    ).to(device)

    # Use evaluation mode
    fusion_model.eval()
    with torch.no_grad():
        combined_embeddings = fusion_model(seq_tensor, hyp_fused_tensor)

    # Save final embeddings
    combined_embeddings = combined_embeddings.cpu().numpy()
    last_save_path = os.path.join(save_path, 'smilesembeddings.npy')
    np.save(last_save_path, combined_embeddings)
    print(f"Saved multimodal embeddings (shape {combined_embeddings.shape}) to {last_save_path}")

    print(f"All processing completed!")

    # Final cleanup
    del fusion_model, hyp_fused_tensor, seq_tensor
    torch.cuda.empty_cache()
    gc.collect()


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default="../data/Davis/", help="Dataset directory path")
#     parser.add_argument('--save_path', type=str, default="../dataset/Davis/", help="Dataset directory path")
#     parser.add_argument('--dataset_name', type=str, default="Davis", help="Dataset name")
#     parser.add_argument('--device', type=str, default="cuda:8", help="Computing device")
#     args = parser.parse_args()
#
#     print(f"Using device: {args.device}")
#     print(f"Processing dataset: {args.dataset}")
#     print(f"Save location after dataset processing: {args.dataset}")
#
#
#     try:
#         get_smiles_embeddings(args.dataset,args.dataset_name,args.save_path, args.device)
#     except Exception as e:
#         print(f"Main function execution error: {str(e)}")
#         import traceback
#
#         traceback.print_exc()