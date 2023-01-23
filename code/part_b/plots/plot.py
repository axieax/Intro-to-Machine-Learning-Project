from ast import Tuple
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Union, Tuple
from os.path import dirname, realpath, join


def plot_1a(
    k_list: List[int], accuracies: Iterable, file_name: Optional[str] = None
) -> None:
    """
    Plot for Part A Q1 (a). Save the plot in the same directory as this file if <file_name> is provided
    """

    plt.figure(figsize=(8, 6))
    plt.title("Validation Accuracies Vs. $k$")
    plt.xlabel("$k$")
    plt.ylabel("Accuracy")
    plt.plot(k_list, accuracies, "--")
    plt.scatter(k_list, accuracies, color="r")
    if file_name:
        dir = dirname(realpath(__file__))
        plt.savefig(join(dir, file_name))
        print(f"Plot saved under {dir}")

    # plt.show()


def plot_2b(
    iterations: Union[int, List[int]],
    LLH: Iterable,
    lr: float,
    file_name: Optional[str] = None,
) -> None:
    """
    :param iterations: Number of iterations or List of each iterations
    :param LLH: Iterable object containing the *positive LLH over iterations
    :param lr: Learning rate used in training

    Plot for Part A Q2 (b).
    Save the plot in the same directory as this file if <file_name> is provided
    """

    if isinstance(iterations, int):
        x = range(iterations)
    else:
        x = iterations

    plt.figure(figsize=(8, 6))
    plt.title(f"Log-likelihood Vs. Iterations \n Learning rate:{lr} n_iter:{len(x)}")
    plt.xlabel("Iteration $i$")
    plt.ylabel("Log-Likelihood")
    plt.plot(x, LLH)
    if file_name:
        dir = dirname(realpath(__file__))
        plt.savefig(join(dir, file_name))
        print(f"Plot saved under {dir}")


def plot_2b_comparison(
    iterations: Union[int, List[int]],
    train_LLH: Iterable,
    validation_LLH: Iterable,
    lr: float,
    file_name: Optional[str] = None,
    compare_avg: bool = False,
) -> None:
    """
    :param iterations: Number of iterations or List of each iterations
    :param lr: Learning rate used in training

    Plot for Part A Q2 (b).
    Save the plot in the same directory as this file if <file_name> is provided
    """

    if isinstance(iterations, int):
        x = range(iterations)
    else:
        x = iterations

    plt.figure(figsize=(8, 6))
    plt.xlabel("Iteration $i$")

    if compare_avg:
        plt.title(
            f"Average Log-likelihood Vs. Iterations \n Learning rate:{lr} n_iter:{len(x)}"
        )
        plt.ylabel("Average Log-Likelihood")

    else:
        plt.title(
            f"Log-likelihood Vs. Iterations \n Learning rate:{lr} n_iter:{len(x)}"
        )
        plt.ylabel("Log-Likelihood")

    plt.plot(x, train_LLH, label="Training Set")
    plt.plot(x, validation_LLH, label="Validation Set")
    plt.legend()
    if file_name:
        dir = dirname(realpath(__file__))
        plt.savefig(join(dir, file_name))
        print(f"Plot saved under {dir}")


def plot_2d(
    theta_list: List[int],
    j1_probs: Iterable,
    j2_probs: Iterable,
    j3_probs: Iterable,
    question_difficulties: Tuple[float],
    file_name: Optional[str] = None,
) -> None:
    """
    Plot for Part A Q2 (d).
    Save the plot in the same directory as this file if <file_name> is provided
    """

    j1_diff, j2_diff, j3_diff = question_difficulties

    plt.figure(figsize=(8, 6))
    plt.title("Prob. of Correct Response Vs. 'Student Ability'")
    plt.xlabel("'Student Ability'")
    plt.ylabel("Probabability of correct response")
    plt.plot(
        theta_list,
        j1_probs,
        "--",
        alpha=0.9,
        label=f"Question $j_1$, difficulty:{j1_diff:.4f}",
    )
    plt.plot(
        theta_list,
        j2_probs,
        "--",
        alpha=0.9,
        label=f"Question $j_2$, difficulty:{j2_diff:.4f}",
    )
    plt.plot(
        theta_list,
        j3_probs,
        "--",
        alpha=0.9,
        label=f"Question $j_3$, difficulty:{j3_diff:.4f}",
    )
    plt.legend()
    # plt.scatter(k_list, accuracies, color="r")
    if file_name:
        dir = dirname(realpath(__file__))
        plt.savefig(join(dir, file_name))
        print(f"Plot saved under {dir}")


def plot_3d(
    num_epoches: int,
    validations: Iterable,
    k_star: int,
    y_axis: str,
    file_name: Optional[str] = None,
) -> None:
    """
    Plot for Part A Q3 (d).
    Save the plot in the same directory as this file if <file_name> is provided
    """
    plt.figure(figsize=(8, 6))
    plt.title(f"{y_axis} Vs. Epoch \n $k^*={k_star}$")
    plt.xlabel("Epoch")
    plt.ylabel(y_axis)
    plt.plot(range(num_epoches), validations)
    if file_name:
        dir = dirname(realpath(__file__))
        plt.savefig(join(dir, file_name))
        print(f"Plot saved under {dir}")


def plot_3d_avg_comparison(
    num_epoches: int,
    training_loss: Iterable,
    validation_loss: Iterable,
    k_star: int,
    file_name: Optional[str] = None,
) -> None:
    """
    Plot for Part A Q3 (d).
    Save the plot in the same directory as this file if <file_name> is provided
    """
    plt.figure(figsize=(8, 6))
    plt.title(f"Average Loss Vs. Epoch \n $k^*={k_star}$")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.plot(range(num_epoches), training_loss, label="Training Set")
    plt.plot(range(num_epoches), validation_loss, label="Validation Set")
    plt.legend()
    if file_name:
        dir = dirname(realpath(__file__))
        plt.savefig(join(dir, file_name))
        print(f"Plot saved under {dir}")


if __name__ == "__main__":

    # k = 30
    # plot_3d(50, range(50), k, "test")
    pass
