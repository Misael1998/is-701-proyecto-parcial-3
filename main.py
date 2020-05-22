from hw_digit_recognizer import HWDigitRecognizer as clf
import pickle

HWD = clf('datasets/mnist_train.csv',
          'datasets/mnist_test.csv')
HWD.train_model()
