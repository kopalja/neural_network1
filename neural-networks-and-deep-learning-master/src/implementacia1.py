
import numpy as np

from theano2 import *



class WordGenerator(object):
    
    key_words = ['dva', 'tri', 'pat', 'sto', 'dom', 'aaa', 'aab', 'aac', 'aad', 'aae', 'aaf']


    def char_to_array(self, ch):
        array_of_byte = ch.encode()
        byte = array_of_byte[0]

        if byte == " ".encode():
            num = 0
        else:
            num = byte - "a".encode()[0] + 1

        out = []
        x = 16
        for i in range(5):
            out.append((num & x) >> 4 - i)
            x //= 2
        return out

    def vectorized_word(self, word):
        result = []
        for ch in word:
            result.extend(self.char_to_array(ch))
        return result
    
    def vectorized_reward(self, word):
        result = np.zeros(5)
        for i in range(len(self.key_words)):
            if (word == self.key_words[i]):
                result[i] = 1
        return result

    def get_training_pair(self):
        r = np.random.randint(0, 1, 1)[0]

        if (r == 0):
            i = np.random.randint(0, len(self.key_words), 1)[0]
            return self.vectorized_word(self.key_words[i]), self.vectorized_reward(self.key_words[i])
        else:
            random_ints = np.random.randint(0, 24, 3)
            word = list("aaa")
            for i in range(3):
                word[i] = chr(random_ints[i] + ord('a'))
            x = "".join(word)
            return self.vectorized_word(x), self.vectorized_reward(x)

    def get_batch(self, s : int):
        data = []
        for i  in range(s):
            data.append(w.get_training_pair())
        data_x = [inpt for inpt, output in data]    
        data_y = [output for inpt, output in data]    
        return tuple([data_x, data_y])  

    def load_data(self):
        training_data = self.get_batch(10000)
        validation_data = self.get_batch(1000)

        def shared(data):
            shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX))
            shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX))
            return shared_x, shared_y
        return [shared(training_data), shared(validation_data)]


theano.config.floatX = 'float32'

w = WordGenerator()

training_data, validation_data = w.load_data()

print(w.get_batch(2))

net = Net(
    [15, 5, 5], 
    training_data, 
    validation_data, 
    CostFunctions().quadratic,
    learning_rate = 1.5, 
    minibatch_size = 10, 
    regulation_param = 0.0
)

net.prepare_and_compile_function()

net.train(epoch = 10)

