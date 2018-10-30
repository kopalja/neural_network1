import numpy as np
import theano
import theano.tensor as T

theano.config.floatX = 'float32'

class CostFunctions:

    @staticmethod    
    def quadratic(output_batch, desire_output_batch):
        def vector_distance(output_vector, desire_output_vector):
            distance, _ = theano.scan(lambda x, y: (x - y) ** 2, [output_vector, desire_output_vector])
            return T.sum(distance)
        cost, _ = theano.scan(
            fn = lambda x, y: vector_distance(x, y),
            sequences = [output_batch, desire_output_batch]
        )
        return T.mean(cost) * 0.5

    @staticmethod
    def crossEntropy(output_batch, desire_output_batch):
        def vector_entropy_distance(output_vector, desire_output_vector):
            distance, _ = theano.scan(lambda x, y: y * T.log(x) + (1.0 - y) * T.log(1.0 - x), [output_vector, desire_output_vector])
            return distance

        distances, _ = theano.scan(
            fn = lambda output_vector, desire_output_vector: vector_entropy_distance(output_vector, desire_output_vector),
            sequences = [output_batch, desire_output_batch]
        )
        return - T.mean(distances)


    @staticmethod
    def probability1(output_batch, desire_output_batch):
        expected_results = T.argmax(desire_output_batch, axis=1)
        return -1 * T.mean(T.log(output_batch)[T.arange(expected_results.shape[0]), expected_results])


    @staticmethod
    def probability(output_batch, desire_output_batch):
        # negative = theano.shared(10.0)
        # expected_results, _ = theano.scan(
        #     fn = lambda desire_output_vector: T.argmax(desire_output_vector), 
        #     sequences = desire_output_batch
        # )
        expected_results = T.argmax(desire_output_batch, axis=1)
        #temp = T.log(output_batch)

        # cost, _ = theano.scan(
        #     fn = lambda output_vector, expected_result: temp[expected_result],
        #     sequences = [temp, expected_results] 
            
        # )
        # return - T.mean(cost)
        return - T.mean(T.log(output_batch)[expected_results])

    @staticmethod
    def probability2(output_batch, desire_output_batch):
        "Return the log-likelihood cost."
        return -1 * T.mean(T.log(output_batch)[T.arange(desire_output_batch.shape[0]), desire_output_batch])
        return -1 * T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])