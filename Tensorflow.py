# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    # plt.scatter(X[:, 0], X[:,1], c=y)
    plt.show()


# plot_decision_boundary (lambda x: logRegression.predict(x))
# plt.title("Logistic Regression")
# plt.show()

# Helper function to predict an output (0 or 1)
def predict(out_act, x, x_raw):

    t = np.zeros(len(x_raw))
    pred = out_act.eval({x: x_raw})
    #print(pred)
    for i in range(len(pred)):
        t[i] = np.argmax(pred[i])
    #print(t)
    return t
    #probs = out_act / np.sum(out_act, axis=1, keepdims=True)

    # t = np.argmax(pred)

    #return pred

def visualize(X, y, out_act,placeholderX):
    #Plot the descion boundary
    #y = predict(model, X)
    y1 = predict(out_act,placeholderX,X)
    #plot_decision_boundary(lambda xarg : predict(out_act,placeholderX,xarg), X, y)
    # plt.title("Decision boundary with 3 hidden layers")
    # plt.show()


def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)

    plt.scatter(X[:,0],X[:,1], c= y)
    plt.show()

    # # scatter plot, dots colored by class value
    # df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    # colors = {0:'red', 1:'blue'}
    # fig, ax = pyplot.subplots()
    # grouped = df.groupby('label')
    # for key, group in grouped:
    #     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    #pyplot.show()

    return X, y

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def loadDataSet():

    X = np.loadtxt("train_X.txt", dtype=int)
    Y = np.loadtxt("train_Y.txt", dtype=int)

    test_X = np.loadtxt("test_X.txt", dtype=int)
    test_Y = np.loadtxt("test_Y.txt", dtype=int)

    return X, test_X, Y, test_Y

def build_model(trainX, trainY, hidden_layers,sess):
    
    trainYOneHot = np.zeros((len(trainX),6), dtype = int)
    trainYOneHot[range(len(trainYOneHot)), trainY] = 1

    n_input = len(trainX[0]) #Number of features
    #print('Feature Length : ', n_input)
    n_hidden = hidden_layers
    n_output = 6  #Number of classes
    
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], mean=0, stddev=1 / np.sqrt(n_input)), name='weights1')
    b1 = tf.Variable(tf.truncated_normal([n_hidden],mean=0, stddev=1 / np.sqrt(n_input)), name='biases1')
    y1 = tf.nn.tanh((tf.matmul(x, W1)+b1), name='activationLayer1')


    #output layer weights and biasies
    Wo = tf.Variable(tf.random_normal([n_hidden, n_output], mean=0, stddev=1/np.sqrt(n_input)), name='weightsOut')
    bo = tf.Variable(tf.random_normal([n_output], mean=0, stddev=1/np.sqrt(n_input)), name='biasesOut')
    #activation function(softmax)
    a = tf.nn.softmax((tf.matmul(y1, Wo) + bo), name='activationOutputLayer')

     #cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a),reduction_indices=[1]))
    #optimizer
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    #compare predicted value from network with the expected value/target
    correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
    #accuracy determination
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    x_raw = np.array(trainX)
    y_raw = np.array(trainY)

    incorrectLabels = 0

    # initialization of all variables
    initial = tf.global_variables_initializer()

    #creating a session
    sess.run(initial)
    for epoch in range(20000):
        k = sess.run(train_step,feed_dict={x: trainX,y: trainYOneHot})
    #sess.close()
    return a, x

# def main():

#     trainX, testX, trainY, testY = loadDataSet()

#     trainYOneHot = np.zeros((len(trainX),6), dtype = int)
#     trainYOneHot[range(len(trainYOneHot)), trainY] = 1

#     n_input = len(trainX[0]) #Number of features
#     #print('Feature Length : ', n_input)
#     n_hidden = 3
#     n_output = 6  #Number of classes

#     # W = { "h1": tf.Variable(tf.ones([n_input, n_hidden]),name="h1"),
#     #         "out": tf.Variable(tf.ones([n_hidden, n_output]))
#     # }

#     # b = { "b1": tf.Variable(tf.zeros([n_hidden])),
#     #         "bout": tf.Variable(tf.zeros([n_output]))
#     # }

#     x = tf.placeholder("float", [None, n_input])
#     y = tf.placeholder("float", [None, n_output])


#     W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], mean=0, stddev=1 / np.sqrt(n_input)), name='weights1')
#     b1 = tf.Variable(tf.truncated_normal([n_hidden],mean=0, stddev=1 / np.sqrt(n_input)), name='biases1')
#     y1 = tf.nn.tanh((tf.matmul(x, W1)+b1), name='activationLayer1')


#     #output layer weights and biasies
#     Wo = tf.Variable(tf.random_normal([n_hidden, n_output], mean=0, stddev=1/np.sqrt(n_input)), name='weightsOut')
#     bo = tf.Variable(tf.random_normal([n_output], mean=0, stddev=1/np.sqrt(n_input)), name='biasesOut')
#     #activation function(softmax)
#     a = tf.nn.softmax((tf.matmul(y1, Wo) + bo), name='activationOutputLayer')

#      #cost function
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a),reduction_indices=[1]))
#     #optimizer
#     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#     #compare predicted value from network with the expected value/target
#     correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
#     #accuracy determination
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

#     x_raw = np.array(trainX)
#     y_raw = np.array(trainY)

#     incorrectLabels = 0

#     # initialization of all variables
#     initial = tf.global_variables_initializer()

#     #creating a session
#     with tf.Session() as sess:
#         sess.run(initial)
#         for epoch in range(20000):
#             for i in range(len(trainX)):
#                 instance = i
#                 k = sess.run(train_step,feed_dict={x: trainX,y: trainYOneHot})
#                 predictLabel = predict(a, x, x_raw[instance])
#                 if predictLabel != trainY[instance]:
#                     incorrectLabels += 1
#         errorRate = incorrectLabels/len(trainX)
#         print('train errorRate : ' , errorRate)

#             # predictedLabel = predict(out_act, x, x_raw[])
#             # # feeding testing data to determine model accuracy
#             # y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: ts_features,keep_prob:1.0})
#             # y_true = sess.run(tf.argmax(ts_labels, 1))

#         visualize(x_raw, y_raw, a, x)

#     sess.close()


# if __name__ == '__main__':
#     main()