import tensorflow as tf
import pickle
import scipy as sp
import time

start_time = time.time()

with open('trainin.pkl', 'rb') as f:
    train_input = pickle.load(f)
with open('trainout.pkl', 'rb') as g:
    train_output = pickle.load(g)
    
with open('testin.pkl', 'rb') as h:
    test_input = pickle.load(h)
with open('testout.pkl', 'rb') as e:
    test_output = pickle.load(e)

#model parameters
sequence_len = 140 #twitter 140 characters
input_unit_dim = 57 #no of total alphabets
no_of_classes = 7 #no of languages

#dimensions for data are [Batch Size, Sequence Length, Input Dimension]
#batch size will be determined at run time
data = tf.placeholder(tf.float32, [None, sequence_len, input_unit_dim])
#Target will hold the training output data which are the correct results that we desire
target = tf.placeholder(tf.float32, [None, no_of_classes])

#NN parameters and initialization
num_hidden = 50 #hidden layers in the NN
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

#unroll the network and pass the data to it and store the output in val
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

# transpose the output to switch batch size with sequence size
val = tf.transpose(val, [1,0,2])
#take the values of output only at seq last input, i.e., in a string of 140 we
#are only interested in output we got at the 140th characters
last = tf.gather(val, int(val.get_shape()[0])-1)

#NN parameters 
'''The dimension of the weights will be num_hidden X number_of_classes. 
Thus on multiplication with the output (val), the resulting dimension 
will be batch_size X number_of_classes which is what we are looking for'''
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

#prediction is the probability score for each class
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
#calculating cross entropy loss that is to be minimized
cross_entropy = -tf.reduce_sum(target*tf.log(prediction))

#optimizing the cross entropy
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

#==============================================================================
'''calculating the error on test data''' 
#==============================================================================
mistakes = tf.not_equal(tf.arg_max(target, 1), tf.arg_max(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
err_list = tf.arg_max(prediction, 1)
#==============================================================================
'''Part-II tensorflow graph execution''' 
#==============================================================================
#initializing all the variables we created
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

#Training on dataset
batch_size = 500
no_of_batches = int(len(train_input)/batch_size)
epoch = 1000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize, {data: inp, target: out})
    print ("Epoch -", str(i))
incorrect = sess.run(error, {data: test_input, target: test_output})
incorr_list = sess.run(err_list, {data: test_input, target: test_output})

print("Epoch {:2d} error {:3.1f}%".format(i+1, 100*incorrect))
Saver = tf.train.Saver()
Saver.save(sess,'S:\Python\Computational_Opt\Comp_Op_Project35')
sess.close()

with open('pred_out.pkl', 'wb') as t:
    pickle.dump(list(incorr_list), t)
end_time = time.time()
print((end_time-start_time)/60.0)









