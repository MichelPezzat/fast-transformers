#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Similar to the corresponding module in fast_transformers.attention, this
module performs all the query, key, value projections and output projections
leaving the implementation of the attention to the inner attention module."""
import torch as t
from torch.nn import Linear, Module

from ....events import EventDispatcher
from ..._utils import check_state
from fast_transformers.ops import Conv1D


class RecurrentAttentionLayer(Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer.

    The only difference with the corresponding module is that this projects
    only one input and then calls the inner attention with the provided
    previous state.

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
        super(RecurrentAttentionLayer, self).__init__()

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

    def forward(self, x, state=None, memory=None):
        """Apply attention to the passed in query/key/value after projecting
        them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            query: (N, D) The tensor containing the queries
            key: (N, D) The tensor containing the keys
            value: (N, D) The tensor containing the values
            state: The state varies depending on the inner attention implementation
            memory: **Deprecated** and replaced by state

        Returns
        -------
            The new value for each query as a tensor of shape (N, D).
        """
        # Normalize the state/memory
        state = check_state(state, memory)

        # Project the queries/keys/values
        #query = self.query_projection(query)
        #key = self.key_projection(key)
        #value = self.value_projection(value)
        x = self.c_attn(x)
        query, key, value = x.chunk(3, dim=2)
        print(query.shape)
        query = t.squeeze(query,1)
        key = t.squeeze(key,1)
        value = t.squeeze(value,1)

        # Reshape them into many heads and compute the attention
        N, D = query.shape
        H = self.n_heads
        new_value, state = self.inner_attention(
            query.view(N, H, -1),
            key.view(N, H, -1),
            value.view(N, H, -1),
            state
        )
        new_value = new_value.view(N, -1)

        # Project the output and return
        return self.c_proj(new_value), state
