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
        pass

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
