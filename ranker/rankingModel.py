from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
torch.autograd.set_detect_anomaly(True)







class DiGCN_Inception_Block_Ranking(RankingGNNBase):
    r"""The ranking model adapted from the
    `Digraph Inception Convolutional Networks"
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **embedding_dim** (int) - Embedding dimension.
        * **Fiedler_layer_num** (int, optional) - The number of single Filder calculation layers, default 3.
        * **alpha** (float, optional) - (Initial) learning rate for the Fiedler step, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
        * **initial_score** (torch.FloatTensor, optional) - Initial guess of scores, default None.
        * **sigma** (float, optionial) - (Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.
        * **kwargs (optional): Additional arguments of
            :class:`RankingGNNBase`.
    """

    def __init__(self, num_features: int, dropout: float,
                 embedding_dim: int, Fiedler_layer_num: int = 3, alpha: float = 0.01,
                 trainable_alpha: bool = False, initial_score: Optional[torch.FloatTensor] = None, prob_dim: int = 5,
                 sigma: float = 1.0, **kwargs):
        super(DiGCN_Inception_Block_Ranking, self).__init__(embedding_dim, Fiedler_layer_num, alpha,
                                                            trainable_alpha, initial_score, prob_dim, sigma, **kwargs)
        self.ib1 = InceptionBlock(num_features, embedding_dim)
        self.ib2 = InceptionBlock(embedding_dim, embedding_dim)
        self._dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()

    def forward(self, \
                edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor], \
                edge_weight_tuple: Tuple[torch.FloatTensor, torch.FloatTensor], \
                features: torch.FloatTensor, ) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * edge_index_tuple (PyTorch LongTensor) - Tuple of edge indices.
            * edge_weight_tuple (PyTorch FloatTensor, optional) - Tuple of edge weights corresponding to edge indices.
            * features (PyTorch FloatTensor) - Node features.

        Return types:
            * z (PyTorch FloatTensor) - Embedding matrix.
        """
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0 + x1 + x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0 + x1 + x2
        self.z = x.clone()

        return self.z