from sklearn.impute import KNNImputer
from utils import *

from plots import plot


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    new_valid_data = dict(
        # HACK: swap user_id and question_id due to matrix tranpose
        user_id=valid_data["question_id"],
        question_id=valid_data["user_id"],
        is_correct=valid_data["is_correct"],
    )
    acc = sparse_matrix_evaluate(new_valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    print(f"----- USER-BASED -----")
    ks = [1, 6, 11, 16, 21, 26]
    acc = []
    for k in ks:
        acc.append(knn_impute_by_user(sparse_matrix, val_data, k))

    best_k_ind = np.argmax(acc)
    k_star = ks[best_k_ind]
    print(f"k* = {k_star}")
    print(f"Test Accuracy: {knn_impute_by_user(sparse_matrix, test_data, k_star)}")

    plot.plot_1a(ks, acc, file_name='q1-user-validation.png')

    print(f"----- ITEM-BASED -----")
    acc = []
    for k in ks:
        acc.append(knn_impute_by_item(sparse_matrix, val_data, k))

    best_k_ind = np.argmax(acc)
    k_star = ks[best_k_ind]
    print(f"k* = {k_star}")
    print(f"Test Accuracy: {knn_impute_by_user(sparse_matrix, test_data, k_star)}")

    plot.plot_1a(ks, acc, file_name='q1-item-validation.png')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
