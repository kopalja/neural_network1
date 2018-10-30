
import numpy





class Test(object):
    x = 10
    def __init__(self, x):
        x = [1, 2, 3]
        self.change(x.copy())
        print(x)

    def change(self, x):
        x[1] = 32

t = Test(3)




# x = T.matrix()
# y = T.matrix()

# result = crossEntropy(x, y)
# foo = theano.function(
#     [x, y],
#     result,
#     allow_input_downcast = True
# )

# x = [[0.1, 0.1, 0.1], [0.1, 0.2, 0.3]]
# y = [[0.3, 0.4, 0.6], [0.5, 0, 0]]

# x = [[1, 1, 1], [1, 1, 1]]
# y = [[1, 1, 9], [2, 2, 4]]

# print(foo(x, y))