"""
Implements a two-layer neural network classifier in PyTorch.
WARNING: you SHOULD NOT use `.to()` or `.cuda()` in each implementation block.
"""
import random
from typing import Callable, Dict, List, Optional

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from two_layer_net.py!")


# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialization of the model. Weights are initialized to small random
        values and biases are initialized to zero. Weights and biases are
        stored in the variable self.params, which is a dictionary with the
        following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(
            hidden_size, dtype=dtype, device=device
        )
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(
            output_size, dtype=dtype, device=device
        )

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    The first stage of our neural network implementation:
    Run the forward pass of the network to compute the hidden layer features
    and classification scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None
    ###########################################################################
    # TODO: Perform the forward pass, computing the class scores for input.   #
    # Store the result in the scores variable, which should be an tensor of   #
    # shape (N, C). You are NOT allowed to use torch.relu and torch.nn ops.   #
    # Hint: torch.clamp                                                       #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)
    matmul = X.mm(W1)+b1
    hidden = torch.clamp(matmul, min=0)
    scores = hidden.mm(W2)+b2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size.

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i]
      is an integer in the range 0 <= y[i] < C. This parameter is optional;
      if it is not passed then we only return scores, and if it is passed,
      then we instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those
      parameters with respect to the loss function; has the same keys as
      params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None
    ###########################################################################
    # TODO: Compute the loss, based on the results from nn_forward_pass.      #
    # This should include both the data loss and L2 regularization for W1 and #
    # W2. Store the result in the variable loss, which should be a scalar.    #
    # Use the softmax classifier loss. When you implement the regularization  #
    # over W, please DO NOT multiply the regularization term by 1/2           #
    # (no coefficient). If you are not careful here, it is easy to run into   #
    # numeric instability. (see "Softmax Classifier - Practical issues:       #
    # numeric stability" in the assignment on Linear Classifiers.)            #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)

    idx_row = torch.arange(N)
    idx_col = y
    new_idx = [idx_row, idx_col]
    correct_score = scores[new_idx].view(-1,1)

    C = scores.max(dim=1)[0].view(-1,1)

    summation = torch.exp(scores + C)
    summation = summation.sum(dim=1).view(-1,1)

    loss = (-1) * correct_score + (-1) * C + torch.log(summation)
    loss = torch.sum(loss)
    loss /= N
    loss += reg * torch.sum(W1*W1)
    loss += reg * torch.sum(W2*W2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    # Backward pass: compute gradients
    grads = {}
    ###########################################################################
    # TODO: Compute the backward pass by computing the derivatives of the     #
    # weights and biases. Store the results in the grads dictionary.          #
    # For example, grads['W1'] should store the gradient on W1, and be a      #
    # tensor of same size. Note that you did not multiply the regularization  #
    # term by 1/2 (no coefficient) above, so the gradient for this term       #
    # should have a scale correspondingly.                                    #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)

    # Getting dW2
    numer = torch.exp(scores)
    denom = torch.exp(scores).sum(dim=1).view(-1,1)

    prob = numer/denom
    prob[new_idx] = prob[new_idx] - 1

    dW2 = h1.t().mm(prob)
    dW2 /= N
    dW2 += reg * 2 * W2
    grads['W2'] = dW2

    # Getting db2
    db2 = prob.sum(dim=0)
    grads['b2'] = db2 / N

    # Getting dW1
    dh1 = prob.mm(W2.t())
    check = (h1>0)
    dh1 = dh1 * check

    grads['W1'] = X.t().mm(dh1)
    grads['W1'] /= N
    grads['W1'] += reg * 2 * W1

    # Getting db1
    grads['b1'] = dh1.sum(dim=0)
    grads['b1'] /= N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, grads


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    ###########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in       #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)    #
    # and y_batch should have shape (batch_size,)                             #
    #                                                                         #
    # Hint: torch.randint; you may want to borrow the device from X.          #
    #       Note that you already implemented this in linear_classifier.py;   #
    #       you can simply copy-paste what you implemented there.             #
    ###########################################################################
    # Replace "pass" statement with your code
    idx = torch.randint(0,num_train, (batch_size,))

    X_batch = X[idx]
    y_batch = y[idx]
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return X_batch, y_batch


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss
          with respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor of shape (N,) giving training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        # sample_batch function is implemented above
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        #######################################################################
        # TODO: Use the gradients in the grads dictionary to update the       #
        # parameters of the network (stored in the dictionary params)         #
        # using stochastic gradient descent. You'll need to use the gradients #
        # stored in the grads dictionary defined above.                       #
        #######################################################################
        # Replace "pass" with your code (do not modify this line)
        params['W1'] = params['W1'] - learning_rate * grads['W1'] 
        params['b1'] = params['b1'] - learning_rate * grads['b1'] 
        params['W2'] = params['W2'] - learning_rate * grads['W2'] 
        params['b2'] = params['b2'] - learning_rate * grads['b2'] 
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each
      of the elements of X. For all i, y_pred[i] = c means that X[i] is
      predicted to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)
    score = loss_func(params, X)
    y_pred = torch.argmax(score, dim=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


def nn_get_search_params():
    """
    Return candidate hyperparameters for a TwoLayerNet model.
    You should provide at least two param for each, and total grid search
    combinations should be less than 256. If not, it will take
    too much time to train on such hyperparameter combinations.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
    """
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []
    ###########################################################################
    # TODO: Add your own hyperparameter lists.                                #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)
    learning_rates = [1e-2, 1e-1, 1]
    hidden_sizes = [8, 32, 64, 128]
    regularization_strengths = [1e-3, 1e-1]
    learning_rate_decays = [1.0, 0.9, 0.8]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )


def find_best_net(
    data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable
):
    """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return
    value of ".train()" operation in best_stat and the validation accuracy of
    the trained best model in best_val_acc. Your hyperparameters should be
    received from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                   (e.g., nn_get_search_params)
                                   that gives
                                   (learning_rates, hidden_sizes,
                                    regularization_strengths,
                                    learning_rate_decays)
                                   You should get hyperparameters from
                                   get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                           (['X_train', 'y_train'], batch_size, learning_rate,
                           learning_rate_decay, reg)
                           for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
    """

    best_net = None
    best_stat = None
    best_val_acc = 0.0

    ###########################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best    #
    # trained model in best_net.                                              #
    #                                                                         #
    # To help debug your network, it may help to use visualizations similar   #
    # to the ones we used above; these visualizations will have significant   #
    # qualitative differences from the ones we saw above for the poorly tuned #
    # network.                                                                #
    #                                                                         #
    # Tweaking hyperparameters by hand can be fun, but you might find it      #
    # useful to write code to sweep through possible combinations of          #
    # hyperparameters automatically like we did on the previous exercises.    #
    #                                                                         #
    # Hint: You can adjust `num_iters` and `batch_size` as well.              #
    # Also, you can `import itertools` if you find it useful. For example,    #
    # itertools.product allows you to iterate over multiple lists of          #
    # variables using one for-loop.                                           #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)
    
    learning_rate, hidden_size, reg, learning_decay  = get_param_set_fn()
    num_models = len(learning_rate)*len(hidden_size)*len(reg)*len(learning_decay)

    i = 0
    best_val_acc = -1.0   
    best_net = None 
    num_iters = 3000
    batch_size = 200

    input_size = data_dict['X_train'].size()[1]
    output_size = torch.max(data_dict['y_train'])+1

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']

    for lr in learning_rate:
      for hs in hidden_size:
        for rs in reg:
          for lrd in learning_decay:

            i += 1
            print('Training TwoLayerNet %d / %d with lr=%e, hs=%e, rs=%e and lrd=%e'
                  % (i, num_models, lr, hs, rs, lrd))
                        
            cand_net = TwoLayerNet(input_size, hs, output_size)

            cand_stat = cand_net.train(X_train, y_train, X_val, y_val, lr, lrd, rs, num_iters, batch_size)

            y_val_pred = cand_net.predict(X_val)
            cand_val_acc = 100.0 * (y_val == y_val_pred).double().mean().item()

            if cand_val_acc > best_val_acc:
              best_val_acc = cand_val_acc
              best_net = cand_net
              best_stat = cand_stat

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_net, best_stat, best_val_acc
