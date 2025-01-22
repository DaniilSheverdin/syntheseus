"""Inference wrapper for the RootAligned model.

Paper: https://arxiv.org/abs/2203.11444
Code: https://github.com/otori-bird/retrosynthesis

The original RootAligned code is released under the MIT license.
Parts of this file are based on code from the GitHub repository above.
"""

import argparse
import math
import multiprocessing
import random
import warnings
from collections import defaultdict
from typing import Any, List, Optional, Sequence

import yaml
from rdkit import Chem

from syntheseus.interface.models import ReactionType
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference.base import ExternalReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards, process_raw_smiles_outputs_forwards,
)


class RootAlignedModel(ExternalReactionModel):
    def is_forward(self) -> bool:
        return self._is_forward

    def __init__(
        self,
        *args,
        is_forward: bool = False,
        num_augmentations: int = 20,
        probability_from_score_temperature: Optional[float] = 2.0,
        **kwargs,
    ) -> None:
        """Initializes the RootAligned model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.pt` file
        - `model_dir` contains the config as the only `*.yml` file
        """
        self._is_forward = is_forward
        super().__init__(*args, **kwargs)

        # Parse arguments for calling external functions from `root_aligned/OpenNMT.py`.
        with open(get_unique_file_in_dir(self.model_dir, pattern="*.yml"), "r") as f:
            opt_from_config = yaml.safe_load(f)

        import torch

        opt = argparse.Namespace()
        for key, value in opt_from_config.items():
            setattr(opt, key, value)

        opt.models = [get_unique_file_in_dir(self.model_dir, pattern="*.pt")]
        opt.output = "/dev/null"
        opt.gpu = -1 if self.device == "cpu" else torch.device(self.device).index

        setattr(opt, "synthon", False)

        from root_aligned import score

        score.opt = opt

        # Import external functions from `root_aligned/OpenNMT.py`.
        from onmt.translate.translator import build_translator
        from onmt.utils.parse import ArgumentParser

        ArgumentParser.validate_translate_opts(opt)

        self.translator = build_translator(opt, report_score=False)
        self.num_augmentations = num_augmentations
        self.probability_from_score_temperature = probability_from_score_temperature
        self.beam_size = opt.beam_size

    def get_parameters(self):
        """Return the model parameters."""
        return self.translator.model.parameters()

    def _mols_to_batch(self, inputs) -> List[bytes]:
        """Map `Molecule`s into SMILES bytes."""
        from root_aligned.score import smi_tokenizer


        if self.is_forward():
            result = []
            for augmented_reactants in inputs:
                s = smi_tokenizer(".".join(molecule.smiles for molecule in augmented_reactants))
                result.append(bytes(s + "\n", "utf-8"))
            return result
        else:
            # Example outcome: b'C C ( = O ) c 1 c c c 2 c ( c c n 2 C ( = O ) O C ( C ) ( C ) C ) c 1\n'.
            return [bytes(smi_tokenizer(input.smiles) + "\n", "utf-8") for input in inputs]

    def _build_kwargs_from_scores(self, scores: List[float]) -> List[ReactionMetaData]:
        """Compute kwargs to save in the predictions given raw scores from the RootAligned model.

        The scores we get from the model cannot be directly interpreted as a (log) probability.
        In general, the model produces an array of `[num_augmentations, beam_size]` predictions, and
        computes the score of a given prediction `p` as `total_rr - best_pos * 1e8` where:
            `total_rr = sum_{(j, k) such that prediction[j, k] = p} (1 / (k + 1))`
            `best_pos = min_{(j, k) such that prediction[j, k] = p} k`
        In other words, the predictions are ranked first on how early they appear during beam search
        for any augmentation (`best_pos`), with ties broken by how much they repeat (`total_rr`).
        See `compute_rank` in the RootAlign repository for details.

        This function recovers `total_rr` and `best_pos`, puts them into metadata, and also computes
        a different combined score which is supposed to be better-behaved (smaller range of values).
        """
        import torch

        # Make sure the scores are sorted as expected.
        for score, next_score in zip(scores, scores[1:]):
            assert score >= next_score

        # Maximum possible value `total_rr` could have.
        max_possible_total_rr = self.num_augmentations * sum(
            1.0 / (k + 1) for k in range(self.beam_size)
        )

        kwargs_list: List[ReactionMetaData] = []
        for score in scores:
            best_pos = -math.floor(score / 1e8)
            total_rr = score + best_pos * 1e8

            assert 0 <= best_pos < self.beam_size
            assert 0.0 < total_rr <= max_possible_total_rr

            new_score = total_rr - (best_pos + 1) * max_possible_total_rr
            assert new_score <= 0.0

            kwargs_list.append(
                {  # type: ignore[typeddict-unknown-key]
                    "original_score": score,
                    "best_pos": best_pos,
                    "total_rr": total_rr,
                    "score": new_score,
                }
            )

        # Make sure the new scores produce the same ranking.
        for kwargs, next_kwargs in zip(kwargs_list, kwargs_list[1:]):
            assert kwargs["score"] >= next_kwargs["score"]

        if self.probability_from_score_temperature is not None:
            scaled_scores = [
                self.probability_from_score_temperature * kwargs["score"] / max_possible_total_rr
                for kwargs in kwargs_list
            ]
            probabilities = torch.nn.functional.softmax(torch.as_tensor(scaled_scores), dim=-1)

            for kwargs, probability in zip(kwargs_list, probabilities.numpy().tolist()):
                kwargs["probability"] = probability

        return kwargs_list

    def _get_reactions(
        self, inputs, num_results: int, random_augmentation=False
    ) -> List[Sequence[ReactionType]]:
        # Step 1: Perform data augmentation.
        if self.is_forward():
            augmented_inputs = self._perform_augmentation_forward(inputs[0])
        else:
            augmented_inputs = self._perform_augmentation_retro(inputs, random_augmentation)

        # Step 2: Map from `Molecule`s to SMILES bytes to align with `root_aligned/OpenNMT.py`.
        augmented_batch = self._mols_to_batch(augmented_inputs)

        # Step 3: Translate.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")

            _, augmented_predictions = self.translator.translate(
                src=augmented_batch,
                src_feats=defaultdict(list),
                tgt=None,
                batch_size=2048,
                batch_type="tokens",
                attn_debug=False,
                align_debug=False,
            )  # shape: `[data_size x augmentation_size, beam_size]`

        # Step 4: Unravel and canonicalize.
        lines = []  # shape: `[data_size x augmentation_size x beam_size]`
        for i in range(len(augmented_predictions)):
            for j in range(len(augmented_predictions[i])):
                lines.append(augmented_predictions[i][j].replace(" ", ""))

        from root_aligned.score import canonicalize_smiles_clear_map

        raw_predictions = []
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        raw_predictions = pool.map(
            func=canonicalize_smiles_clear_map, iterable=lines
        )  # Canonicalize reactants and modify illegal reactants into empty strings.
        pool.close()
        pool.join()

        # From `[data_size x augmentation_size x beam_size]` to `[data_size, augmentation_size, beam_size]`.
        predictions: List[List[Any]] = [
            [[] for _ in range(self.num_augmentations)] for _ in range(len(inputs))
        ]

        for i, prediction in enumerate(raw_predictions):
            predictions[i // (self.beam_size * self.num_augmentations)][
                i % (self.beam_size * self.num_augmentations) // self.beam_size
            ].append(prediction)

        # Step 5: Rank legal reactants from all augmentations and beams.
        ranked_results = []  # shape: `[data_size, augmentation_size x beam_size]`
        ranked_scores = []

        from root_aligned.score import compute_rank

        for i in range(len(predictions)):
            rank, _ = compute_rank(predictions[i])
            rank = list(zip(rank.keys(), rank.values()))
            rank.sort(key=lambda x: x[1], reverse=True)
            rank = rank[:num_results]  # Truncate to `num_results` results.
            ranked_results.append([item[0][0] for item in rank])  # Output reactant SMILES.
            ranked_scores.append([item[1] for item in rank])  # Output scores used for ranking.

        if self.is_forward():
            return [
                process_raw_smiles_outputs_forwards(
                    input, outputs, self._build_kwargs_from_scores(scores)
                )
                for input, outputs, scores in zip(inputs, ranked_results, ranked_scores)
            ]
        return [
            process_raw_smiles_outputs_backwards(
                input, outputs, self._build_kwargs_from_scores(scores)
            )
            for input, outputs, scores in zip(inputs, ranked_results, ranked_scores)
        ]

    def _perform_augmentation_retro(self, inputs, random_augmentation):
        augmented_inputs = []
        if random_augmentation:
            for input in inputs:
                augmented_inputs.append(input)
                for i in range(self.num_augmentations - 1):
                    randomized_smi = Chem.MolToSmiles(input.rdkit_mol, doRandom=True)
                    randomized_mol = Molecule(smiles=randomized_smi, canonicalize=False)
                    augmented_inputs.append(randomized_mol)
        else:
            from root_aligned.preprocessing.generate_PtoR_data import clear_map_canonical_smiles

            for input in inputs:
                product_atom_map_numbers = [i + 1 for i in range(input.rdkit_mol.GetNumAtoms())]
                max_times = len(product_atom_map_numbers)
                product_roots = [-1]
                times = min(self.num_augmentations, max_times)
                if times < self.num_augmentations:  # times = max_times
                    product_roots.extend(product_atom_map_numbers)
                    product_roots.extend(
                        random.choices(product_roots, k=self.num_augmentations - len(product_roots))
                    )
                else:  # times = num_augmentations
                    while len(product_roots) < times:
                        product_roots.append(random.sample(product_atom_map_numbers, 1)[0])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == self.num_augmentations
                for k in range(times):
                    pro_root_atom_map = product_roots[k]
                    pro_root = pro_root_atom_map - 1
                    if pro_root_atom_map <= 0:
                        pro_root = -1
                    pro_smi = clear_map_canonical_smiles(
                        input.smiles, canonical=True, root=pro_root
                    )
                    randomized_mol = Molecule(smiles=pro_smi, canonicalize=False)
                    augmented_inputs.append(randomized_mol)

        assert len(augmented_inputs) == len(inputs) * self.num_augmentations

        return augmented_inputs

    def _perform_augmentation_forward(self, inputs):
        from root_aligned.preprocessing.generate_RtoP_data import clear_map_canonical_smiles

        augmented_reactants_by_iteration = []

        reactant_atom_map_numbers_list = [
            [i + 1 for i in range(input.rdkit_mol.GetNumAtoms())] for input in inputs
        ]
        max_times_list = [len(product_atom_map_numbers) for product_atom_map_numbers in reactant_atom_map_numbers_list]

        min_times = min(self.num_augmentations, max(max_times_list))

        for k in range(min_times):
            iteration_augmentations = []

            for idx, input in enumerate(inputs):
                reactant_atom_map_numbers = reactant_atom_map_numbers_list[idx]
                max_times = max_times_list[idx]
                times = min(self.num_augmentations, max_times)
                reactant_roots = [-1]

                # Генерация массива корней (augmented points) для текущего реагента
                if times < self.num_augmentations:
                    reactant_roots.extend(reactant_atom_map_numbers)
                    reactant_roots.extend(
                        random.choices(reactant_roots, k=self.num_augmentations - len(reactant_roots))
                    )
                else:
                    while len(reactant_roots) < self.num_augmentations:
                        reactant_roots.append(random.sample(reactant_atom_map_numbers, 1)[0])
                        if reactant_roots[-1] in reactant_roots[:-1]:
                            reactant_roots.pop()

                assert len(reactant_roots) == self.num_augmentations

                # Для текущей итерации выбираем корень
                pro_root_atom_map = reactant_roots[k]
                pro_root = pro_root_atom_map - 1
                if pro_root_atom_map <= 0:
                    pro_root = -1

                # Канонизация SMILES и создание молекулы
                pro_smi = clear_map_canonical_smiles(
                    input.smiles, canonical=True, root=pro_root
                )
                randomized_mol = Molecule(smiles=pro_smi, canonicalize=False)

                iteration_augmentations.append(randomized_mol)

            augmented_reactants_by_iteration.append(iteration_augmentations)

        return augmented_reactants_by_iteration
