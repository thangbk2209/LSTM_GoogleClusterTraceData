import numpy as np

__name__="Initialization"
def xavier_initiliazer(shape=None,uniform=True):
        seed = np.random.randint(1,1E3)
        random_engine = np.random.RandomState(seed=seed)
        if(uniform==True):
            range_init = np.sqrt(6.0/(shape[0]+shape[1]))
            return random_engine.uniform(-range_init,range_init,size=(shape[0],shape[1]))
        else:
            stdv = np.sqrt(3.0/(shape[0]+shape[1]))
            return random_engine.normal(0.0,scale=stdv,size=(shape[0],shape[1]))
def initialize_param(weights_layer,bias_layer,uniform_init=True):
    W = []
    b = []
    for weight in weights_layer:
        W.append(xavier_initiliazer(weight,uniform_init))
    for bias in bias_layer:
        b.append(np.zeros(bias))
    return W,b
def construct_candidate(neural_shape,uniform_init=True):
    weights_layer = [(neural_shape[t-1],neural_shape[t]) for t in range(1,len(neural_shape))]
    bias_layer = [neural_shape[t] for t in range(1,len(neural_shape))]
    W,b = initialize_param(weights_layer,bias_layer,uniform_init)
    param_network = zip(W,b)
    total_weights = []
    for param in param_network:
        temp_weight = np.concatenate([param[0].flatten(),param[1].flatten()])
        total_weights.append(temp_weight)
    return np.concatenate([total_weights[0],total_weights[1]]).flatten()
def construct_solution(number_of_solutions,neural_shape,uniform_init=True):
    return np.array([construct_candidate(neural_shape,uniform_init) for t in range(number_of_solutions)])
