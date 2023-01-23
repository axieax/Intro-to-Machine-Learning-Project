import numpy as np

from utils import load_train_csv, load_valid_csv, load_public_test_csv
import resample
from item_response_list import irt, sigmoid


def evaluate(data, thetas, betas):
    """ Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param thetas: list[Vector]
    :param betas: list[Vector]
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        count_p = 0
        for t, b in zip(thetas, betas):
            u = data["user_id"][i]
            x = (t[u] - b[q]).sum()
            p_a = sigmoid(x)
            count_p += int(p_a >= 0.5)

        pred.append(count_p >= 2)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    train_data_1 = resample.resample(train_data, seed=3111)
    train_data_2 = resample.resample(train_data, seed=3112)
    train_data_3 = resample.resample(train_data, seed=3113)

    lr, n_epoch = 0.005, 100
    theta_1, beta_1, val_acc_lst_1 = irt(train_data_1, val_data, lr, n_epoch)
    theta_2, beta_2, val_acc_lst_2 = irt(train_data_2, val_data, lr, n_epoch)
    theta_3, beta_3, val_acc_lst_3 = irt(train_data_3, val_data, lr, n_epoch)

    thetas, betas = [theta_1, theta_2, theta_3], [beta_1, beta_2, beta_3]

    print('\n')
    print(f'Validation Accuracy: {evaluate(val_data, thetas, betas)}')
    print(f'Test Accuracy: {evaluate(test_data, thetas, betas)}')


if __name__ == '__main__':
    main()
