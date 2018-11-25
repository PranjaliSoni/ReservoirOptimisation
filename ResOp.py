
from numpy import exp,array,random,dot
import xlrd
from scipy.special import expit
class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((4,1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return expit(-x)

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_inputs, training_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_inputs, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

   # print("Random starting synaptic weights: ")
   # print(neural_network.synaptic_weights)

    loc =(r"C:\Users\hp\Downloads\FINAL_DATAf.xls")
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    nrows=sheet.nrows
    ncols=sheet.ncols
    print(ncols)
    set_inputs = [[ncols]*nrows for x in range(nrows)]
    set_outputs = [nrows for x in range(nrows )]
    for i in  range(nrows):
        if i!=0:
            set_inputs[i][0]=  sheet.cell_value(i,0)
            set_inputs[i][1] = sheet.cell_value(i, 1)
            x=sheet.cell_value(i, 2)
            set_outputs[i-1] = x
            set_inputs[i][2] = sheet.cell_value(i, 3)
            set_inputs[i][3] = sheet.cell_value(i, 4)

    print("hello")
    #The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    #neural_network.train(training_set_inputs, training_set_outputs, 10000)
    print("vf")
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0,3])))
