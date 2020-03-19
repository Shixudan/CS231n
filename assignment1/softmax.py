from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classed = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores_ = scores - max(scores) #为了数值的稳定性
        scores_ = np.exp(scores_) #取指数
        softmax_sum = np.sum(scores_)
        scores_ /= softmax_sum
        loss -= np.log(scores_[y[i]])  # 得到交叉熵
        #计算梯度
        for j in range(num_classed):
            if j != y[i]:
                dW[:, j] += scores_[j] * X[i]
            else:
                dW[:, j] -= (1 - scores_[j]) * X[i]
    loss /= X.shape[0] #求平均
    dW /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score = X.dot(W)
    score -= np.max(score, axis=1, keepdims=True)
    score = np.exp(score)
    score /= np.sum(score, axis=1, keepdims=True) #除以分母得到softmax

    loss = -np.sum(np.log(score[np.arange(X.shape[0]), y]))
    loss /= X.shape[0]

    ds = np.copy(score)
    ds[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(ds) # X*W=S  求导链式法则
    dW /= X.shape[0]

    loss += reg * np.sum(W *W)
    dW += 2 * reg *W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
