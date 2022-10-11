"""
File for all the utilities.
"""
from collections import defaultdict
from collections import namedtuple
from ctypes import Union
from typing import Tuple
import numpy as np
import tensorflow as tf


BEST_SCORES = namedtuple(
    'BEST_SCORES', 'best_loss_x best_loss_y_x best_loss_y best_loss_x_y'
)


def return_all_scores(
    final_scores: defaultdict(dict)
):
    """
    This helper takes a default dict that has the run number as keys
    and a dit of random restarts as values. It returns a dict with the run
    number as the key and a list of the random restart scores.
    """
    all_loss_x = defaultdict(list)
    all_loss_y_x = defaultdict(list)
    all_loss_y = defaultdict(list)
    all_loss_x_y = defaultdict(list)

    for run_idx in list(final_scores.keys()):
        run_rr_scores = final_scores[run_idx]
        for rr_idx in list(run_rr_scores.keys()):
            current_rr_score = run_rr_scores[rr_idx]
            all_loss_x[run_idx].append(current_rr_score.loss_x)
            all_loss_y_x[run_idx].append(current_rr_score.loss_y_x)
            all_loss_y[run_idx].append(current_rr_score.loss_y)
            all_loss_x_y[run_idx].append(current_rr_score.loss_x_y)

    return (all_loss_x, all_loss_y_x, all_loss_y, all_loss_x_y)


def return_best_causal_scores(
    all_loss_x: defaultdict(list),
    all_loss_y_x: defaultdict(list),
    all_loss_y: defaultdict(list),
    all_loss_x_y: defaultdict(list)
) -> dict:
    """
    This takes dictionaries of all the scores and returns a dictionary with
    the run number as a key and the best scores as values - named tuple of type
    BEST_SCORES.
    """
    all_runs = list(all_loss_x.keys())
    best_scores = {}

    for run_idx in all_runs:
        best_x = min(all_loss_x[run_idx])
        best_y_x = min(all_loss_y_x[run_idx])
        best_y = min(all_loss_y[run_idx])
        best_x_y = min(all_loss_x_y[run_idx])

        best_causal_score = BEST_SCORES(
            best_x, best_y_x, best_y, best_x_y
        )
        best_scores[run_idx] = best_causal_score

    return best_scores


def get_correct(best_scores: dict, targets: list):
    """
    Get the correct and wrong idx.
    """
    correct_idx = []
    wrong_idx = []
    for run_idx in list(best_scores.keys()):
        current_scores = best_scores[run_idx]
        current_target = targets[run_idx]
        x_causes_y_score = current_scores.best_loss_x + current_scores.best_loss_y_x
        y_causes_x_score = current_scores.best_loss_y + current_scores.best_loss_x_y
        if current_target > 0:
            if x_causes_y_score < y_causes_x_score:
                correct_idx.append(run_idx)
            elif y_causes_x_score < x_causes_y_score:
                wrong_idx.append(run_idx)
            else:
                tf.print(f"Can't decide for run {run_idx}")
        if current_target < 0:
            if x_causes_y_score < y_causes_x_score:
                wrong_idx.append(run_idx)
            elif y_causes_x_score < x_causes_y_score:
                correct_idx.append(run_idx)
            else:
                tf.print(f"Can't decide for run {run_idx}")
    return correct_idx, wrong_idx

