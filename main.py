from hw_digit_recognizer import HWDigitRecognizer as clf
import pickle

HWD = clf('autograder_data/mnist_train_0.01sampled.csv',
          'autograder_data/mnist_test_0.01sampled.csv')
HWD.train_model()
