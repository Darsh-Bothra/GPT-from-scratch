import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined_list = positive + negative
        st = sorted({word for sentence in combined_list for word in sentence.split()})

        w2i = {word: i+1 for i, word in enumerate(st)}

        encode = [torch.tensor([w2i[w] for w in s.split()]) for s in combined_list]

        return nn.utils.rnn.pad_sequence(encode, batch_first=True)
        
        
