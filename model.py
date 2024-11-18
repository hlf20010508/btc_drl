import torch
from torch import nn


class GRUPPO(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, action_dim: int, num_layers: int
    ):
        super(GRUPPO, self).__init__()

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state):
        out, hidden_state = self.gru(x, hidden_state)
        out = out[:, -1, :]
        action_probs = torch.softmax(self.actor(out), dim=-1)
        value = self.critic(out)
        return action_probs, value, hidden_state
