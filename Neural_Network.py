import numpy as np
from sklearn.datasets import make_moons
import sklearn.linear_model 
from matplotlib import pyplot as plt
import warnings


class neuralNetwork():
    

    #Gradient descent paramenters
    eps = 0.01 #Learning rate for gradient descent
    regularization = 0.01 #Regularization strength 

def generate_Dataset():
    #Generate dataset
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    return X, y

    #Training the logistic regression classifier
    # logRegression = sklearn.linear_model.LogisticRegressionCV()
    # logRegression.fit(X, y)

#Plot the decision boundary

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# plot_decision_boundary (lambda x: logRegression.predict(x))
# plt.title("Logistic Regression")
# plt.show()

#Loss function
def calculate(model, X, y):
    #Implementing neural network
    num_examples = len(X)

    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    #Forward propogation to calculate prediction
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    #Increasing number of hidden layers
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)

    z3 = a2.dot(W3) + b3
    exp_score = np.exp(z3)
    probs =  exp_score / np.sum(exp_score, axis=1, keepdims=True)

    #Calculating loss
    correct_logProb = - np.log(probs[range(num_examples),y])
    data_loss = np.sum(correct_logProb)
    # Add regulatization term to loss (optional)
    data_loss += neuralNetwork.regularization/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2) + np.sum(np.square(W3))))
    return 1./num_examples * data_loss 

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    y = np.argmax(probs, axis=1)
    return y


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_h_dim1, nn_h_dim2, num_passes=5000, print_loss=False):

    nn_input_dim = 144 #input layer dimensionality
    nn_output_dim = 6 #output layer dimensionality

    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_h_dim1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_h_dim1))
    W2 = np.random.randn(nn_h_dim1, nn_h_dim2) / np.sqrt(nn_h_dim2)
    b2 = np.zeros((1, nn_h_dim2))
    W3 = np.random.randn(nn_h_dim2, nn_output_dim) / np.sqrt(nn_h_dim2)
    b3 = np.zeros((1, nn_output_dim))

    #This is what we returned in the end
    model = {}

    #Gradient descent
    for i in range(0, num_passes):

        #Forward propogation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)

        z2 = a1.dot(W2) + b2
        a2 = np.tanh(z2)

        z3 = a2.dot(W3) + b3
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        #Back propogation
        delta4 = probs
        for i in range(num_examples):
            for j in range(0,6):
                delta4[i,j] = y[i,j]-1
        #delta4[range(num_examples, 6), y] -= 1
        
        dW3 = (a2.T).dot(delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        
        delta3 = delta4.dot(W3.T) * (1 - np.power(a2, 2))
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1- np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        #Regularization terms
        dW3 += neuralNetwork.regularization * W3
        dW2 += neuralNetwork.regularization * W2
        dW1 += neuralNetwork.regularization * W1

        #Gradient descent parameter update
        W1 += - neuralNetwork.eps * dW1
        b1 += - neuralNetwork.eps * db1
        W2 += - neuralNetwork.eps * dW2
        b2 += - neuralNetwork.eps * db2
        W3 += - neuralNetwork.eps * dW3
        b3 += - neuralNetwork.eps * db3

        #Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        # if print_loss and i % 1000 == 0:
        #     print ("Loss after iteration %i: %f" %(i, calculate(model, X, y)))

    return model


def visualize(X, y, model):
    #Plot the descion boundary
    #y = predict(model, X)
    y1 = predict(model, X)
    print(y1)
    # plt.show(X, y1)
    #plot_decision_boundary(lambda x : predict(model,x), X, y1)
    # plt.title("Decision boundary with 3 hidden layers")    
    # plt.show()


warnings.filterwarnings("always")

# def main():
#     neuralNetwork()
#     X, y = generate_Dataset()
#     #3 hidden layers
#     model = build_model(X, y, 3,3,print_loss=True)
#     visualize(X,y, model)

# if __name__=="__main__":
#     main()