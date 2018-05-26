import numpy as np
class ParameterNoise(object):
    def apply(self, model):
        pass

class NormalParameterNoise(ParameterNoise):
    def __init__(self, sigma):
        self.sigma = sigma

    def apply(self, params):
        # TODO: behavior of model.parameters() on networks with shared convolution
        #print('paramtype', type(params))
        '''
        for p in params:
            print(type(params))
            shape = tuple(p.data.shape)
            print("shape", shape)
            noise = np.random.normal(0, self.sigma, size=shape)
            p.data = p.data + noise
        '''

    def __repr__(self):
        return 'NormalParameterNoise(sigma={})'.format(self.sigma)