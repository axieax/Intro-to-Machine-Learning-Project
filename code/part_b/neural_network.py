import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from config import *
from plots import plot
from torch.autograd import Variable
from utils import *


def load_data(base_path="../data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100, meta_fields=0):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        :param meta_fields: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question + meta_fields, k + meta_fields)
        self.g2 = nn.Linear(k + meta_fields, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        g_out = torch.sigmoid(self.g(inputs))
        g2_out = torch.sigmoid(self.g2(g_out))
        h_out = torch.sigmoid(self.h(g2_out))

        out = h_out
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(
    model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, k, student_meta
):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param k: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses, avg_train_losses, valid_losses, avg_valid_losses = [], [], [], []
    n_train_obs = torch.numel(train_data) - torch.sum(torch.isnan(train_data))

    valid_acc_hist = []
    max_hist_len = 5

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            meta_fields = tuple(
                Variable(torch.Tensor([[student_meta[user_id][field]]]))
                for field in student_meta_fields
            )
            target = inputs.clone()
            inputs = torch.cat((inputs,) + meta_fields, dim=1)

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            # nan_mask = np.concatenate((nan_mask, [[False]]), axis=1)

            target[0][nan_mask] = output[0][nan_mask]

            reg = model.get_weight_norm() * lamb / 2.0
            loss = torch.sum((output - target) ** 2.0)
            train_loss += loss.item()
            loss += reg
            loss.backward()

            optimizer.step()

        valid_acc, valid_loss = evaluate(
            model, zero_train_data, valid_data, student_meta
        )
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        avg_train_losses.append(train_loss / n_train_obs)
        avg_valid_losses.append(valid_loss / len(valid_data["user_id"]))
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t "
            "Valid Acc: {}".format(epoch, train_loss, valid_acc)
        )

        # # early stopping, used for hyperparameter tuning, disabled to avoid issues with variation
        # if len(valid_acc_hist) < max_hist_len:
        #     valid_acc_hist.append(valid_acc)
        # elif all(valid_acc_hist[i] > valid_acc_hist[i + 1] for i in range(max_hist_len - 1)):
        #     print(f'EARLY STOP @ {epoch=}\tValid Acc History: {valid_acc_hist}')
        #     break  # stop training early
        # else:
        #     for i in range(max_hist_len - 1):
        #         valid_acc_hist[i] = valid_acc_hist[i + 1]
        #     valid_acc_hist[-1] = valid_acc

    train_losses.extend([float('inf')] * (num_epoch - epoch - 1))
    valid_losses.extend([float('inf')] * (num_epoch - epoch - 1))
    avg_train_losses.extend([float('inf')] * (num_epoch - epoch - 1))
    avg_valid_losses.extend([float('inf')] * (num_epoch - epoch - 1))

    plot.plot_3d(
        num_epoch, train_losses, k, "Training Loss", file_name="partb-train-loss.png"
    )
    plot.plot_3d(
        num_epoch, valid_losses, k, "Validation Loss", file_name="partb-valid-loss.png"
    )

    plot.plot_3d_avg_comparison(
        num_epoch, avg_train_losses, avg_valid_losses, k, file_name="partb-avg-loss.png"
    )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data, student_meta):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    valid_loss = 0.0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        meta_fields = tuple(
            Variable(torch.Tensor([[student_meta[u][field]]]))
            for field in student_meta_fields
        )
        inputs = torch.cat((inputs,) + meta_fields, dim=1)
        output = model(inputs)

        target = valid_data["is_correct"][i]
        pred_tensor = output[0][valid_data["question_id"][i]]
        guess = pred_tensor.item() >= 0.5
        if guess == target:
            correct += 1
        total += 1

        valid_loss += torch.sum((pred_tensor - target) ** 2.0).item()

    return correct / float(total), valid_loss


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    student_meta = load_student_data("../data")

    # Set model and optimization hyperparameters. (from `config.py`)
    model = AutoEncoder(
        train_matrix.shape[1], k=k, meta_fields=len(student_meta_fields)
    )

    print(f"{k=} {lr=} {num_epoch=} {lamb=}")

    train(
        model,
        lr,
        lamb,
        train_matrix,
        zero_train_matrix,
        valid_data,
        num_epoch,
        k,
        student_meta,
    )

    print(
        f"Test Accuracy: {evaluate(model, zero_train_matrix, test_data, student_meta)[0]}"
    )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
