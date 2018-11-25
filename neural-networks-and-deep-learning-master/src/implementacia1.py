
import numpy as np

from theano2 import *



class WordGenerator(object):
    
    key_words = ['stolick', 'trinast', 'patstos', 'stoosem', 'domovom']



    def char_to_array(self, ch):
        array_of_byte = ch.encode()
        byte = array_of_byte[0]

        if byte == " ".encode():
            num = 0
        else:
            num = byte - "a".encode()[0] + 1

        out = []
        x = 16
        for i in range(len(self.key_words)):
            out.append((num & x) >> 4 - i)
            x //= 2
        return out

    def vectorized_word(self, word):
        result = []
        for ch in word:
            result.extend(self.char_to_array(ch))
        return result
    
    def vectorized_reward(self, word):
        result = np.zeros(len(self.key_words))
        for i in range(len(self.key_words)):
            if (word == self.key_words[i]):
                result[i] = 1
        return result

    def get_training_pair(self, x):
        r = np.random.randint(self.down, self.up, 1)[0]

        if (r > 50):
            i = np.random.randint(0, len(self.key_words), 1)[0]
            return self.vectorized_word(self.key_words[i]), self.vectorized_reward(self.key_words[i])
        else:
            random_ints = np.random.randint(0, 24, 7)
            word = list("aaaaaaa")
            for i in range(7):
                word[i] = chr(random_ints[i] + ord('a'))
            x = "".join(word)
            return self.vectorized_word(x), self.vectorized_reward(x)

    def get_batch(self, s : int, x):
        data = []
        for i  in range(s):
            data.append(w.get_training_pair(x))
        data_x = [inpt for inpt, output in data]    
        data_y = [output for inpt, output in data]    
        return tuple([data_x, data_y])  

    def load_data(self):
        self.down = 0
        self.up = 100
        training_data = self.get_batch(1000, 100)
        self.down = 60
        self.up = 100
        validation_data1 = self.get_batch(100, 50)
        self.down = 0
        self.up = 20
        validation_data2 = self.get_batch(100, 50)

        def shared(data):
            shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX))
            shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX))
            return shared_x, shared_y
        return [shared(training_data), shared(validation_data1), shared(validation_data2)]


theano.config.floatX = 'float32'

w = WordGenerator()

training_data, validation_data1, validation_data2  = w.load_data()


net = Net(
    [5 * 7, 6, 5], 
    training_data, 
    validation_data1,
    validation_data2, 
    CostFunctions().quadratic,
    learning_rate = 0.9, 
    minibatch_size = 10, 
    regulation_param = 0.0
)

net.prepare_and_compile_function()

net.train(epoch = 100)

