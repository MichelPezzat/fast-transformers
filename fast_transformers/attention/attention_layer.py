#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""The base attention layer performs all the query key value projections and
output projections leaving the implementation of the attention to the inner
attention module.

The transformer layers, however, are agnostic of the attention implementation
and any layer that implements the same interface can substitute for the
attention layer.
"""

from torch.nn import Linear, Module
from fast_transformers.ops import Conv1D

from ..events import EventDispatcher, QKVEvent


class AttentionLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher="", zero_out=False, init_scale=1.0):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        #self.query_projection = Linear(d_model, d_keys * n_heads)
        #self.key_projection = Linear(d_model, d_keys * n_heads)
        #self.value_projection = Linear(d_model, d_values * n_heads)
        self.c_attn = Conv1D(d_model, d_keys * 3, init_scale=init_scale)
        #self.out_projection = Linear(d_values * n_heads, d_model)
        self.c_proj = Conv1D(d_values, d_model, zero_out, init_scale=init_scale)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = x.shape
        H = self.n_heads

        # Project the queries/keys/values
        #queries = self.query_projection(queries).view(N, L, H, -1)
        #keys = self.key_projection(keys).view(N, S, H, -1)
        #values = self.value_projection(values).view(N, S, H, -1)

        x = self.c_attn(x)



        queries, keys, values = x.chunk(3, dim=2)
        queries = queries.view(N, L, H, -1)
        keys = keys.view(N, L, H, -1)
        values = values.view(N, L, H, -1)

        print(queries.shape)


        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.c_proj(new_values)
