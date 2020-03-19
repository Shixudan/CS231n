from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)   #计算得到给定W下的线性liner预测值
        correct_class_score = scores[y[i]] #正确label对应的分数
        for j in range(num_classes):
            if j == y[i]:
                continue
            #真实值与预测值之间的差距
            #为什么margin大于0的情况下才进行更新dw
            #解释：svm线性分类器只关心正确分类和预测分类差距在某个阈值之间的数据，同时开始计算损失值
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                #添加dw的变化，为什么使用X[i]
                #解释：因为当前关于W的梯度是xi。因为是batch训练，加起来后再求平均，再乘learningrate更新weights
                #这里都是分类出错的数据，朝着梯度上升的方向更新
                dW[:, j] += X[i].T
                #分类正确的数据，也需更新w，朝着梯度下降的方向更新
                dW[:, y[i]] += -X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.（特别注意乘0.5）
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
    # correct_class_score = scores[np.arange(num_train), y] #1 * N
    # correct_class_score = np.reshape(correct_class_score, (num_train, 1)) #N * 1
    #对于loss，间隔margins中把j = yi项赋值为0，margins小于0处赋值为0，对剩下的元素求和取平均，最后加上正则项。
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins)  / num_train + 0.5 * reg * np.sum(W * W)

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #对于dw，首先间隔margins小于0处赋值为0，其余赋值为1。
    # dWj = X.T.dot(margins)。对应的dWyi = -X.T.dot(margins)。
    DW_mat = np.zeros((num_train, num_classes))
    DW_mat[margins > 0] = 1
    DW_mat[range(num_train), list(y)] = 0
    DW_mat[range(num_train), list(y)] = -np.sum(DW_mat, axis=1)

    dW = X.T.dot(DW_mat)
    dW = dW / num_train + reg * W
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
