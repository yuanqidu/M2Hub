"""
Codes borrowed from EGNN (https://github.com/vgsatorras/egnn)
"""


from m2models.models.egnn.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn


import torch
import torch.nn as nn

from torch_scatter import scatter

from m2models.common.registry import registry
from m2models.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from m2models.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from m2models.models.base import BaseModel

class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    # def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
    #     row, col = edge_index
    #     trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
    #     agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
    #     coord += agg*self.coords_weight
    #     return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, num_atoms=None):
        row, col = edge_index
        row = row.to(torch.long)
        col = col.to(torch.long)
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


@registry.register_model("egnn")
class EGNN(BaseModel):
    def __init__(self,
                 num_atoms,
                 bond_feat_dim,
                 num_targets,
                 otf_graph=False,
                 use_pbc=True,
                 regress_forces=True,
                 output_dim=0,
                 in_edge_nf=0, 
                 hidden_nf=128, 
                 # device='cpu', 
                 readout="mean",
                 act_fn=nn.SiLU(), 
                 n_layers=4, 
                 coords_weight=1.0, 
                 attention=False, 
                 node_attr=1,
                 embeddings="qmof",
    ):
        super(EGNN, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.num_targets = num_targets
        self.readout = readout

        self.hidden_nf = hidden_nf
        # self.device = device
        self.n_layers = n_layers

        # Get atom embeddings
        if embeddings == "khot":
            embeddings = KHOT_EMBEDDINGS
        elif embeddings == "qmof":
            embeddings = QMOF_KHOT_EMBEDDINGS
        else:
            raise ValueError(
                'embedding mnust be either "khot" for original CGCNN K-hot elemental embeddings or "qmof" for QMOF K-hot elemental embeddings'
            )
        self.embedding = torch.zeros(100, len(embeddings[1]))
        in_node_nf = len(embeddings[1])
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])

        ### Encoder
        self.embedding_fc = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        if output_dim != 0:
            self.num_targets = output_dim
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, self.num_targets))
        # self.to(self.device)

        self.dtype = torch.float32



    def _forward(self, h0, x, edges, edge_attr, num_atoms):
        h = self.embedding_fc(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=h0, num_atoms=num_atoms)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr,
                                                      node_attr=None, num_atoms=num_atoms)

        h = self.node_dec(h)

        index = torch.arange(len(num_atoms)).to(num_atoms.device).repeat_interleave(num_atoms)
        hsum = scatter(h, index, dim=0, reduce=self.readout)
        pred = self.graph_dec(hsum)
        return pred.squeeze(1)

    def forward(self, data):

        (
            edge_index,
            distances,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        data.edge_index = edge_index

        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)

        nodes = torch.stack([torch.tensor(self.embedding[int(at.item()-1)]) for at in data.atomic_numbers]).to(data.pos.device, self.dtype)
        atom_positions = data.pos.to(data.pos.device, self.dtype)
        edges = data.edge_index.to(data.pos.device, self.dtype)

        if self.regress_forces:
            data.pos.requires_grad_(True)

        energy = self._forward(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, num_atoms=data.natoms)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy
