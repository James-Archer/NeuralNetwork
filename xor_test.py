from neuralNetwork import *

xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_outputs = [0, 1, 1, 0]

def f(outputs, expectedOutputs):
    dif = 0
    for i in range(0,len(outputs)):
        dif += (outputs[i][0] - expectedOutputs[i])**2
    return (1 - dif)

netTemp = Network()
netTemp.createNetwork([2,1], [1])
netTemp.populateNetwork()   

a = Trainer(netTemp, 100, f, xor_inputs, xor_outputs)
best = a.run(0.99999, 100, selectionMethod = "Dynamic")
print('done')
