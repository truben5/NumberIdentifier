import mnist_loader
import Network

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

net = Network.Network([784,50,10])
net.SGD(training_data, 30, 10, 1.75, test_data=test_data)