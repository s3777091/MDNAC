from __future__ import annotations


class MDCLinearAttentionCache:
    def __init__(self, n_layers: int) -> None:
        self.conv_states = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.has_previous_state = False

    def reset(self) -> None:
        for index in range(len(self.conv_states)):
            self.conv_states[index] = None
            self.recurrent_states[index] = None
        self.has_previous_state = False


class MDCDecoderCache:
    def __init__(self, n_layers: int) -> None:
        self.cache = [None] * n_layers
        self.linear_cache = MDCLinearAttentionCache(n_layers)

    def get(self, layer_idx: int):
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value) -> None:
        self.cache[layer_idx] = value

    def reset(self) -> None:
        for index in range(len(self.cache)):
            self.cache[index] = None
        self.linear_cache.reset()
