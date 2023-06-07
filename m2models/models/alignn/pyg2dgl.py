"""
Codes borrowed from ALIGNN (https://github.com/usnistgov/alignn/tree/main)
"""
 
from typing import Union, Any
import torch_geometric
try:
    import torch
    import dgl
except Exception as exp:
    print("dgl/torch/tqdm is not installed.", exp)
    pass


# copied from most current pyg, since the version running in ocp-models doesn't 
# have to_dgl()
def to_dgl(
    data: Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
    object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The data object.

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
        >>> x = torch.randn(5, 3)
        >>> edge_attr = torch.randn(6, 2)
        >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes=5, num_edges=6,
            ndata_schemes={'x': Scheme(shape=(3,))}
            edata_schemes={'edge_attr': Scheme(shape=(2, ))})

        >>> data = HeteroData()
        >>> data['paper'].x = torch.randn(5, 3)
        >>> data['author'].x = torch.ones(5, 3)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        >>> data['author', 'cites', 'paper'].edge_index = edge_index
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes={'author': 5, 'paper': 5},
            num_edges={('author', 'cites', 'paper'): 5},
            metagraph=[('author', 'paper', 'cites')])
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()
        g = dgl.graph((row, col), num_nodes=data.natoms)
        g.ndata['atomic_numbers'] = data.atomic_numbers
        g.ndata['tags'] = data.tags
        g.ndata['pos'] = data.pos

        # g.edata['distances'] = data.distances
        g.edata['cell_offsets'] = data.cell_offsets
        # print (g)
        # for attr in data.node_attrs():
        #     g.ndata[attr] = data[attr]
        # for attr in data.edge_attrs():
        #     if attr in ['edge_index', 'adj_t']:
        #         continue
        #     g.edata[attr] = data[attr]

        return g

    # if isinstance(data, HeteroData):
    #     data_dict = {}
    #     for edge_type, store in data.edge_items():
    #         if store.get('edge_index') is not None:
    #             row, col = store.edge_index
    #         else:
    #             row, col, _ = store['adj_t'].t().coo()

    #         data_dict[edge_type] = (row, col)

    #     g = dgl.heterograph(data_dict)

    #     for node_type, store in data.node_items():
    #         for attr, value in store.items():
    #             g.nodes[node_type].data[attr] = value

    #     for edge_type, store in data.edge_items():
    #         for attr, value in store.items():
    #             if attr in ['edge_index', 'adj_t']:
    #                 continue
    #             g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


def compute_bond_cosines(edges):
    # copied from ALIGNN
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}

def _convert2dgl(geometric_data, dist_vect_r, dist):
    g = to_dgl(geometric_data)
    # print (dist_vect_r.shape, geometric_data, g)
    # adding 'r' into edata for compute bond cosines
    g.edata["r"] = dist_vect_r
    g.edata["distances"] = dist
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    return g, lg

def convert2dgl(geometric_batch_data, distance_vec, distances):
    device = geometric_batch_data.pos.device
    g_list = []
    lg_list = []
    geo_data_list = geometric_batch_data.to_data_list()
    # split 'r'/distance vec into data list
    dist_vec_slice_idx = geometric_batch_data._slice_dict['edge_index']
    # ignoring first splits since they are empty
    distance_vec_list = list(torch.tensor_split(distance_vec, dist_vec_slice_idx)[1:-1])
    distances_list = list(torch.tensor_split(distances, dist_vec_slice_idx)[1:-1])
    for geo_data, dist_vec, dist in zip(geo_data_list, distance_vec_list, distances_list):   
        g, lg = _convert2dgl(geo_data, dist_vec, dist)
        g_list.append(g)
        lg_list.append(lg)
        
    return (dgl.batch(g_list).to(device), dgl.batch(lg_list).to(device))
 