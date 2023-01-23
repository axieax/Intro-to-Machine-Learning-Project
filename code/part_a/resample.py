import numpy as np


def resample(train_data: dict, seed: int = None) -> dict:

    out = {"user_id": [], "question_id": [], "is_correct": []}

    user_id = train_data["user_id"]
    question_id = train_data["question_id"]
    is_correct = train_data["is_correct"]

    N = len(is_correct)

    if seed is not None:
        np.random.seed(seed)

    for i in range(len(is_correct)):
        idx = np.random.randint(N)

        out["user_id"].append(user_id[idx])
        out["question_id"].append(question_id[idx])
        out["is_correct"].append(is_correct[idx])

    return out
