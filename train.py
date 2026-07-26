import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # Go through all the epochs
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ids = torch.randint(len(data) - context_length, (batch_size, ))
            
            X = torch.stack([data[id: id+context_length] for id in ids])
            y = torch.stack([data[id+1: id+context_length+1] for id in ids])

            logits = model(X)
            B, T, C = logits.shape

            loss = F.cross_entropy(logits.reshape(B*T, C), y.reshape(B*T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)