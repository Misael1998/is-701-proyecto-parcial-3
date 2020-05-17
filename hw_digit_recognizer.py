import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


class HWDigitRecognizer:

    def __init__(self, train_filename, test_filename):
        """El método init leerá los datasets con extensión ".csv" cuyas ubicaciones
        son recibidas mediante los paramentros <train_filename> y <test_filename>. 
        Los usará para crear las matrices de X_train, X_test, Y_train y Y_test con 
        las dimensiones adecuadas y normalización de acuerdo a lo definido en clase 
        para un problema de clasificación multiclase resuelto media una única red 
        neuronal.
        """
        train_file = pd.read_csv(train_filename)
        test_file = pd.read_csv(test_filename)

        y_train_tmp = train_file.label.to_numpy()
        y_test_tmp = test_file.label.to_numpy()

        y_train = []
        y_test = []

        for i in y_train_tmp:
            a = np.zeros(10)
            a[i] = 1
            y_train.append(a)

        for i in y_test_tmp:
            a = np.zeros(10)
            a[i] = 1
            y_test.append(a)

        y_train = np.asanyarray(y_train).T
        y_test = np.asanyarray(y_test).T

        x_train = train_file.drop('label', axis=1).to_numpy().T
        x_test = test_file.drop('label', axis=1).to_numpy().T

        self.X_train = x_train/255
        self.X_test = x_test/255
        self.Y_train = y_train
        self.Y_test = y_test

    def train_model(self):
        """
        Entrena complementamente una red neuronal con múltiples capas, utilizando 
        la función de activación RELU en las primeras L-1 capas y la función Softmax
        en la última capa de tamaño 10 para realizar clasificación multiclase. 

        Retorna una tupla cuyo primer elemento es un diccionario que contiene los 
        parámetros W y b de todas las capas del modelo con esta estructura:

        { "W1": ?, "b1": ?, ... , "WL": ?, "bL": ?}

        donde los signos <?> son sustituidos por arreglos de numpy con los valores 
        respectivos. El valor de L será elegido por el estudiante mediante experimentación. 
        El segundo elemento a retornar es una lista con los costos obtenidos durante 
        el entrenamiento cada 100 iteraciones.

        Por razones de eficiencia el autograder revisará su programa usando un 
        dataset más pequeño que el que se proporciona (usted puede hacer lo 
        mismo para sus pruebas iniciales). Pero una vez entregado su proyecto se 
        harán pruebas con el dataset completo, por lo que el diccionario que 
        retorna este método con los resultados del entrenamiento con una precisión 
        mayor al 95% en los datos de prueba debe ser entregado junto con este 
        archivo completado.

        Para entregar dicho diccionario deberá guardarlo como un archivo usando el 
        módulo "pickle" con el nombre y extensión "all_params.dict", este archivo 
        deberá estar ubicado en el mismo directorio donde se encuentra  el archivo 
        actual: hw_digit_recognizer.py. El autograder validará que este archivo esté 
        presente y tendra las claves correctas, pero la revisión de la precisión se 
        hará por el docente después de la entrega. La estructura del archivo será la 
        siguiente:

        {
          "model_params": { "W1": ?, "b1": ?, ... , "WL": ?, "bL": ?},
          "layer_dims": [30, ..., 10],  
          "learning_rate": 0.001,
          "num_iterations": 1500,
          "costs": [0.2356, 0.1945, ... 0.00345]
        }

        <model_params>, es un diccionario que contiene los valores de las L * 2  matrices
        de parámetros del modelo (numpy array) desde W1 y b1, hasta WL y bL, el valor
        de L será elegido por el estudiante.
        <layer_dims>, es una lista con las dimensiones de las L+1 capas del modelo, 
        incluyendo la capa de entrada.
        <learning_rate>, el valor del ritmo de entrenamiento.
        <num_iterations>, el número de iteraciones que se usó.
        <costs>, una lista con los costos obtenidos cada 100 iteraciones.
        """

        dict_params = {}
        params = {}
        costs = []
        np.random.seed(1)
        layer_dims = [784, 600, 7, 2, 3]
        num_iterations = 2500
        learning_rate = 0.0075
        print_cost=True

        params = initialize_parameters_deep(layer_dims)

        X = self.X_train
        Y = self.Y_train

        for i in range(0, num_iterations):
            AL, caches = L_model_forward(X, params)

            cost = compute_cost(AL, Y)

            grads = L_model_backward(AL, Y, caches)

            params = update_parameters(params, grads, learning_rate)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        return params, dict_params

    def predict(self, X, model_params):
        """
        Retorna una matriz de predicciones de <(1,m)> con valores entre 0 y 9 que 
        representan las etiquetas para el dataset X de tamaño <(n,m)>.

        <model_params> contiene un diccionario con los parámetros <w> y <b> de cada 
        uno de los clasificadores tal como se explica en la documentación del método <train_model>.
        """
        pass

    def get_datasets(self):
        """Retorna un diccionario con los datasets preprocesados con los datos y 
        dimensiones que se usaron para el entrenamiento

        d = { "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test
        }
        """
        d = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "Y_train": self.Y_train,
            "Y_test": self.Y_test
        }

        return d

# Funciones de ayuda:

def sigmoid(z):

    s = 1/(1 + np.exp(-z))
    return s

def relu(x):

    m = np.maximum(0,x)
    return m

def softmax(x):

    expo = np.exp(x)
    expo_sum = np.sum(expo)
    r = expo/expo_sum
    return r

def initialize_parameters_deep(layer_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def linear_forward(A, W, b):

    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))

    return Z


def linear_activation_forward(A_prev, W, b, activation):

    Z = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = sigmoid(Z)

    elif activation == "relu":
        A = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (Z, A_prev, W)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        W = parameters["W" +str(l)]
        b = parameters["b" +str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)

    A_prev = A
    b = parameters["b" +str(L)]
    W = parameters["W" +str(L)]
    AL, cache = linear_activation_forward(A_prev, W, b, "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost

def linear_bacward(dZ, cache):

    Z, A_prev, W = cachem = A_prev.shape[1]
    dW = 1/m * dZ.dot(A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = (W.T).dot(dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    Z, A_prev, W = cache
    
    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        
    elif activation == "sigmoid":
        s = sigmoid(Z)
        dZ = dA * s * (1 - s)
        
    dA_prev, dW, db = linear_backward(dZ, cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL =-(Y/AL)+(1-Y)/(1-AL)

    current_cache=caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
             linear_activation_backward(dAL, current_cache, "sigmoid")
    
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = \
             linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    
    return parameters

#Pruebas:

train_f = pd.read_csv('autograder_data/mnist_train_0.01sampled.csv')
x_temp = train_f.drop('label', axis=1).to_numpy().T
X = x_temp/255

n_0 = X.shape[0]
n_1 = X.shape[1]

#parameters = initialize_parameters_deep([784, 600, 7, 2, 3])
parameters = initialize_parameters_deep([5, 4, 7, 2, 3])

#ALP, caches = L_model_forward(X, parameters)
#print("AL = " + str(ALP))
#print("Length of caches list = " + str(len(caches)))

# print(X.shape)
print(parameters)
#print(X.shape[1])
