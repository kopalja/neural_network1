
import numpy as np



class WordGenerator(object):
    
    key_words = ['dva', 'tri', 'pat', 'sto', 'dom']


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

    def get_training_pair(self):
        r = np.random.randint(0, 2, 1)[0]

        if (r == 0):
            i = np.random.randint(0, len(self.key_words), 1)[0]
            return self.vectorized_word(self.key_words[i]), 1
        else:
            random_ints = np.random.randint(0, 24, 3)
            word = list("aaa")
            for i in range(3):
                word[i] = chr(random_ints[i] + ord('a'))
            x = "".join(word)
            for word in self.key_words:
                if (word == x):
                    return self.vectorized_word(x), 1
            return self.vectorized_word(x), 0

w = WordGenerator()
for i  in range(10):
    print(w.get_training_pair()) 