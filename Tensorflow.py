# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)



# Helper function to predict an output (0 or 1)
def predict(out_act, x, x_raw):

    t = np.zeros(len(x_raw))
    pred = out_act.eval({x: x_raw})
    #print(pred)
    for i in range(len(pred)):
        t[i] = np.argmax(pred[i])
    #print(t)
    return t
    
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