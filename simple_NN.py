import numpy as np
#one sample has 4 input (say 4 sensors readings)
#batch of 3 samples at a time
inputs = [[1, 2, 3, 2.5],               #<--sample1
          [2.0, 5.0, -1.0, 2.0],        #<--sample2
          [-1.5, 2.7, 3.3, -0.8]]       #<--sample3

#layer 1
weights = [[0.2, 0.8, -0.5, 1.0],           #<--n1 incomming weights
           [0.5, -0.91, 0.26, -0.5],        #<--n2 incomming weights
           [-0.26, -0.27, 0.17, 0.87]]      #<--n3 incomming weights
biases = [2, 3, 0.5]                            #n1,n2,n3 biases

#layer 2
weights2 = [[0.1, -0.14, 0.5],              #<--n1 incomming weights
           [-0.5, 0.12, -0.33],             #<--n2 incomming weights
           [-0.44, 0.73, -0.13]]            #<--n3 incomming weights
biases2 = [-1, 2, -0.5]                         #n1,n2,n3 biases

weights = np.array(weights).T # matrix transposition
weights2 = np.array(weights2).T # matrix transposition

output1 = np.dot(inputs, weights) + biases      #layer1 output
output2 = np.dot(output1, weights2) + biases2   #layer2 output

print(output2)  #neural network output
