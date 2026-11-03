"""OFFICIAL GRADER: do not modify.

Reads budget.json and scores submission.npz against the hidden test set, which is mounted at grading time.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = json.load(open(os.path.join(HERE, "..", "budget.json")))


def score(submission_path):
    raise RuntimeError("the hidden test set is only mounted at grading time")


if __name__ == "__main__":
    print(f"max_neurons = {BUDGET['max_neurons']}")
