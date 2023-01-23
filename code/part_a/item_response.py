import scipy.sparse

from utils import *

import numpy as np

from plots import plot


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: data matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    if isinstance(data, np.ndarray):
        diff_mat = np.tile(theta, (beta.shape[0], 1)).T - np.tile(beta, (theta.shape[0], 1))
        const = np.log(1 + np.exp(diff_mat))
        log_lklihood = np.nansum(data * diff_mat - const)
    else:
        diff_mat = np.tile(theta, (beta.shape[0], 1)).T - np.tile(beta, (theta.shape[0], 1))
        const = np.log(1 + np.exp(diff_mat))
        log_lklihood = np.nansum(data.multiply(diff_mat) - const)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: data matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff_mat = np.tile(theta, (beta.shape[0], 1)).T - np.tile(beta, (theta.shape[0], 1))
    sig_diff_mat = sigmoid(diff_mat)
    d_theta = np.nansum(data.A - sig_diff_mat, axis=1)
    d_beta = np.nansum(sig_diff_mat - data.A, axis=0)

    theta += lr * d_theta.squeeze()
    beta += lr * d_beta.squeeze()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: data matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(data.shape[0])
    beta = np.zeros(data.shape[1])

    train_nll_lst = []
    val_nll_lst = []

    # SPARSE MATRICES FILL WITH 0, NOT NAN
    # val_matrix = scipy.sparse.csc_matrix((val_data['is_correct'], (val_data['user_id'], val_data['question_id'])))
    val_matrix = np.empty(data.shape) * np.nan
    for u, q, d in zip(val_data['user_id'], val_data['question_id'], val_data['is_correct']):
        val_matrix[u, q] = d

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_nll_lst.append(-train_neg_lld)
        val_nll_lst.append(-val_neg_lld)
        print("NLLK: {:.4f} \t Score: {:.4f} {}".format(train_neg_lld, score, val_neg_lld))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_nll_lst, val_nll_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

# Helper function for Q2 (d)
def _correct_prob(theta, beta):
    """
    Return P(c=1|theta, beta)
    """
    return np.exp(theta - beta) / (1 + np.exp(theta - beta))


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr, n_iter = 0.005, 100
    theta, beta, train_nll_lst, val_nll_lst = irt(sparse_matrix, val_data, lr, n_iter)
    val_acc = evaluate(train_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f'Validation Accuracy: {val_acc} --- Test Accuracy: {test_acc}')

    plot.plot_2b(n_iter, train_nll_lst, lr=lr, file_name='q2a-train-nll.png')
    plot.plot_2b(n_iter, val_nll_lst, lr=lr, file_name='q2a-val-nll.png')
    plot.plot_2b_comparison(n_iter, train_nll_lst, val_nll_lst, lr=lr, file_name='q2a-compare-nll.png')
    plot.plot_2b_comparison(n_iter,
                            list(map(lambda nll: nll / len(train_data['is_correct']), train_nll_lst)),
                            list(map(lambda nll: nll / len(val_data['is_correct']), val_nll_lst)),
                            lr=lr, file_name='q2a-compare-avg-nll.png', compare_avg=True)


    #####################################################################
    # Implement part (d)                                                #
    #####################################################################


    np.random.seed(314)
    j1, j2, j3 = np.random.randint(0, len(beta), 3)     # index of questions
    print(f"Randomly Selected Questions: {j1, j2, j3}")
    j1_difficulty, j2_difficulty, j3_difficulty = beta[j1], beta[j2], beta[j3]

    sorted_theta = theta.copy()
    sorted_theta.sort()

    j1_probs = _correct_prob(sorted_theta, j1_difficulty)
    j2_probs = _correct_prob(sorted_theta, j2_difficulty)
    j3_probs = _correct_prob(sorted_theta, j3_difficulty)

    plot.plot_2d(
        sorted_theta,
        j1_probs, j2_probs, j3_probs,
        (j1_difficulty, j2_difficulty, j3_difficulty),
        "q2d-plot.png")

    # TODO: Interpret negative difficulties of questions

    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
