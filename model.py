import torch
from torch import nn


class GRUPPO(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int,
        dropout=0.2,
    ):
        super(GRUPPO, self).__init__()

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.action_ratio = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state: torch.Tensor):
        # x: (batch_size, seq_len, input_dim)
        # hidden_state: (num_layers, batch_size, hidden_dim)
        # x may not full fill the batch, so we need to slice the hidden state
        out, hidden_state = self.gru(
            x, hidden_state.detach()[:, : x.size(0), :].contiguous()
        )
        out = out[:, -1, :]

        action_probs = torch.softmax(self.actor(out), dim=-1)
        action_ratio = self.sigmoid(self.action_ratio(out))

        value = self.critic(out)
        return action_probs, action_ratio, value, hidden_state
