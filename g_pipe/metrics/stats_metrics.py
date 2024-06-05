import numpy as np
from typing import List, Dict
from jieba.posseg import cut
from collections import defaultdict
from tqdm import tqdm


class PartOfSpeechEntropy:

    def __init__(
        self,
        n_workers=1,
        use_paddle=False,
    ) -> None:
        self._n_workers = n_workers
        self._use_paddle = use_paddle

    def __call__(
        self,
        file_paths: List[str],
        show_progress=True,
    ) -> Dict[str, float]:
        if self._n_workers == 1:
            entropies = list(
                tqdm(
                    map(
                        self._compute_entropy,
                        file_paths,
                    ),
                    total=len(file_paths),
                    disable=not show_progress,
                )
            )
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                entropies = list(
                    tqdm(
                        executor.map(
                            self._compute_entropy,
                            file_paths,
                        ),
                        total=len(file_paths),
                        disable=not show_progress,
                    )
                )

        return dict(zip(file_paths, entropies))

    def _compute_entropy(self, file_path: str) -> float:
        """
        Compute the entropy of the part of speech of the given file path.

        Args:
            text (str): The text to compute the entropy of the part of speech.

        Returns:
            float: The entropy of the part of speech of the text.
        """
        text = open(file_path, "r", encoding="utf-8").read()
        counter = defaultdict(int)
        for word, pos in cut(text, use_paddle=self._use_paddle):
            counter[pos] += 1
        total = sum(counter.values())
        return sum(-count / total * np.log(count / total) for count in counter.values())
