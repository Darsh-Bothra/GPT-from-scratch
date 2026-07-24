from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        split_char = list(corpus)
        ans = []

        for _ in range(num_merges):
            freq = {}

            # Count adjacent token pairs
            for i in range(len(split_char) - 1):
                pair = (split_char[i], split_char[i + 1])
                freq[pair] = freq.get(pair, 0) + 1

            if not freq:
                break

            # Highest frequency, lexicographically smallest tie
            best_pair = sorted(
                freq.items(),
                key=lambda item: (-item[1], item[0])
            )[0][0]

            ans.append([best_pair[0], best_pair[1]])

            # Merge non-overlapping occurrences
            new = []
            j = 0

            while j < len(split_char):
                if j < len(split_char) - 1 and (
                    split_char[j], split_char[j + 1]
                ) == best_pair:
                    new.append(split_char[j] + split_char[j + 1])
                    j += 2
                else:
                    new.append(split_char[j])
                    j += 1

            split_char = new

        return ans