from typing import List, Dict

class Solution:

    def _tokenize(self, text: str, vocab: Dict[str, int]):
        tokens = []
        i = 0
        mxl = len(text)
        
        while i < mxl:
            tkn, buff = "", ""
            for t in text:
                buff += t
                if buff in vocab:
                    tkn = buff
            tokens.append(tkn)
            text = text.removeprefix(tkn)
            i += len(tkn) if tkn else mxl
        
        return tokens
            
        
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans = []
        for num in numbers:
            text = str(num)
            ans.append(self._tokenize(text, vocab))
        
        return ans


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        text = "".join([x for x in text])
        tke = self._tokenize(text, vocab)
        return len(tke)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self._tokenize(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)
