import tensorflow as tf
import pickle
import scipy as sp
import time


class LangClass:
    def __init__(self, hid_layers, seq_len=140, inp_dim=57, no_lan=7, batch=500):
        self.sequence_len = seq_len #twitter 140 characters
        self.input_unit_dim = inp_dim #no of total alphabets
        self.no_of_classes = no_lan #no of languages
        
        self.num_hidden = hid_layers #hidden layers in the NN
        
        self.batch_size = batch
        
        #dimensions for data are [Batch Size, Sequence Length, Input Dimension]
        #batch size will be determined at run time
        self.data = tf.placeholder(tf.float32, [None, self.sequence_len, self.input_unit_dim])
        #Target will hold the training output data which are the correct results that we desire
        self.target = tf.placeholder(tf.float32, [None, self.no_of_classes])
        
    def rnn(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=True)

        #unroll the network and pass the data to it and store the output in val
        self.val, self.state = tf.nn.dynamic_rnn(self.cell, self.data, dtype=tf.float32)

        # transpose the output to switch batch size with sequence size
        self.val = tf.transpose(self.val, [1,0,2])
        #take the values of output only at seq last input, i.e., in a string of 140 we
        #are only interested in output we got at the 140th characters
        self.last = tf.gather(self.val, int(self.val.get_shape()[0])-1)

        #NN parameters 
        '''The dimension of the weights will be num_hidden X number_of_classes. 
        Thus on multiplication with the output (val), the resulting dimension 
        will be batch_size X number_of_classes which is what we are looking for'''
        self.weight = tf.Variable(tf.truncated_normal([self.num_hidden, int(self.target.get_shape()[1])]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))

        #prediction is the probability score for each class
        self.prediction = tf.nn.softmax(tf.matmul(self.last, self.weight) + self.bias)
        #calculating cross entropy loss that is to be minimized
        self.cross_entropy = -tf.reduce_sum(self.target*tf.log(self.prediction))

        #optimizing the cross entropy
        self.optimizer = tf.train.AdamOptimizer()
        self.minimize = self.optimizer.minimize(self.cross_entropy)
        
    def classify(self):
        self.mistakes = tf.not_equal(tf.arg_max(self.target, 1), tf.arg_max(self.prediction, 1))
        self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))
        self.err_list = tf.arg_max(self.prediction, 1)
        
        
    def execution(self):
        #initializing all the variables we created
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        
        #Training on dataset
        no_of_batches = int(len(train_input)/self.batch_size)
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
        