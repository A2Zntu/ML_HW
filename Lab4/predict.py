import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples
    
# You need to return the following variables correctly 
    p = np.zeros(m)

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
    
# add a bias to x
    X = np.hstack((np.ones((X.shape[0], 1)), X))
# calculate the a_2
    a_2 = sigmoid(np.dot(X, np.transpose(Theta1)))
# add a bias to a_2
    a_2 = np.hstack((np.ones((a_2.shape[0], 1)), a_2))
# calculate the output layer
    h = sigmoid(np.dot(a_2, np.transpose(Theta2)))
# indexing the maxium element of the rows; 
# i.e. find the most possible number index   
    p = np.argmax(h, axis = 1) 
# because the index of reorder. eg. 10 in the index 9    
    p = p + 1


# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#
    return p

# =========================================================================
