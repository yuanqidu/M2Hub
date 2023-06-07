"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.

Codes borrowed from ALIGNN (https://github.com/usnistgov/alignn/tree/main)
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from m2models.common.registry import registry
from m2models.common.utils import (
    conditional_grad,
)
from m2models.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from m2models.models.base import BaseModel
from m2models.models.alignn.pyg2dgl import convert2dgl

"""Shared model-building components."""
from typing import Optional

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self, in_features: int, out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)

@registry.register_model("alignn")
class ALIGNN(BaseModel):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        alignn_layers=4,
        gcn_layers=4,
        num_gaussians=80,
        triplet_input_features=40,
        embedding_features=64,
        atom_embedding_size=256,
        output_dim=0,
        link="identity",
        regress_forces=False,
        embeddings="khot",
        cutoff=6.0,
        readout="mean",
        use_pbc=True,
        otf_graph=False,
        ):
        """Initialize class with number of input features, conv layers."""
        super(ALIGNN, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.readout = readout

        # Get CGCNN atom embeddings
        if embeddings == "khot":
            embeddings = KHOT_EMBEDDINGS
        elif embeddings == "qmof":
            embeddings = QMOF_KHOT_EMBEDDINGS
        else:
            raise ValueError(
                'embedding mnust be either "khot" for original CGCNN K-hot elemental embeddings or "qmof" for QMOF K-hot elemental embeddings'
            )
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])

        # adapted from original ALIGNN repo
        self.atom_embedding = MLPLayer(
            len(embeddings[1]), atom_embedding_size)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=num_gaussians,),
            MLPLayer(bond_feat_dim, embedding_features),
            MLPLayer(embedding_features, atom_embedding_size),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=triplet_input_features,
            ),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, atom_embedding_size),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(atom_embedding_size, atom_embedding_size,)
                for idx in range(alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    atom_embedding_size, atom_embedding_size
                )
                for idx in range(gcn_layers)
            ]
        )

        if output_dim != 0:
            self.num_targets = output_dim

        if self.readout == 'mean':
            self.readout = AvgPooling()
        self.fc = nn.Linear(atom_embedding_size, self.num_targets)
        self.link = None
        self.link_name = link
        if link == "identity":
            self.link = lambda x: x
        elif link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif link == "logit":
            self.link = torch.sigmoid

        

    @conditional_grad(torch.enable_grad())
    def _forward(
        self, g
    ):
        """ALIGNN : start with `atom_features`.
        
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        ##############################
        # CHANGED
        ##############################

        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = self.embedding[g.ndata['atomic_numbers'].long() - 1]
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = g.edata.pop("distances")
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out)

    
    def forward(self, data):
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        (
            edge_index,
            distances,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)   
        # # currently ALIGNN can't run when otf_graph is true
        # # because edge_index is computed after batch is created
        # # and therefore data._slice_dict['edge_index'] is None 
        # # A change should follow below:
        # if self.otf_graph:
        #     compute _slice_dict for edge_index manually
        #     data._slice_dict['edge_index'] = edge_index_slice index
        # else:
        data.edge_index = edge_index
        
        g = convert2dgl(data, distance_vec, distances)
        # g[0].edata['distances'] = distances

        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(g)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces[data.pos.shape[0]]
        else:
            return energy