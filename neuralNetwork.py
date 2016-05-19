from math import exp
from random import gauss, gammavariate, choice, uniform
from copy import deepcopy
from time import clock
from neuralNetConstants import *

class Network():

    def __init__(self):

        self.layers = []
        self.inputs = []
        self.outputs = []
        self.synapses = []
        self.neuronCount = 0
        self.fitness = 0
        self.prevFitness = 0

    def addInputs(self, n):
        # Adds n input nodes
        temp = []
        for i in range(0,n):
            temp.append(InputNode())
        self.inputs = temp

    def addOutputs(self, n):
        # Adds n output nodes
        temp = []
        for i in range(0,n):
            temp.append(OutputNode())
        self.outputs = temp

    def addLayer(self, n, index = None):
        # Adds a layer of n nodes. If no second arg is specified, goes to end
        temp = []
        for i in range(0,n):
            temp.append(Neuron())
        if index:
            self.layers.insert(index, temp)
        else:
            self.layers.append(temp)
        self.countNeurons()

    def addSynapse(self, neuronIn, neuronOut):
        # Connects two neurons (or inputs/outputs) with a synapse
        self.synapses.append(Synapse(neuronIn))
        neuronOut.inputSynapses.append(self.synapses[-1])

    def addNeuron(self, layerIndex = None):
        # Inserts a neuron into a layer. If no index is specified, a new layer is added to end
        if layerIndex:
            self.layers[layerIndex].append(Neuron())
        else:
            self.addLayer(1)

    def step(self):
        # Once the inputs are set, this will send them through the network
        for layer in self.layers:
            for neuron in layer:
                neuron.getInputs()
        for output in self.outputs:
            output.getInputs()

    def createNetwork(self, inOut, layers):
        # Sets up the network structure (in/out and neurons)
        self.clearNetwork()
        self.addInputs(inOut[0])
        self.addOutputs(inOut[1])
        for i in layers:
            self.addLayer(i)       

    def populateNetwork(self, synType = "Full", clear = False, completeness = 0.5):
        # Connects the neurons with synapses
        if synType == "Full":
            for inp in self.inputs:
                for neuron in self.layers[0]:
                    self.addSynapse(inp, neuron)

            for i in range(0, len(self.layers)-1):
                for neuron1 in self.layers[i]:
                    for neuron2 in self.layers[i+1]:
                        self.addSynapse(neuron1, neuron2)

            for neuron in self.layers[-1]:
                for output in self.outputs:
                    self.addSynapse(neuron, output)

        elif synType == "Random":

            totalSynapses = int(completeness*self.maxConnections())
            flatlist = []
            for layer in self.layers:
                for neuron in layer:
                    flatlist.append(neuron)
                    
            for inp in self.inputs:
                neuron = choice(flatlist + self.outputs)
                self.addSynapse(inp, neuron)

            for neuron in flatlist:
                if not neuron.inputSynapses:
                    options = self.inputs
                    for layer in self.layers:
                        if neuron in layer:
                            layerIndex = self.layers.index(layer)
                            break
                    for i in range(0, layerIndex):
                        for neuron2 in self.layers[i]:
                            options.append(neuron2)
                    neuron2 = choice(options)
                    self.addSynapse(neuron2, neuron)

            while len(self.synapses) < totalSynapses:
                allChoice = self.inputs + flatlist + self.outputs
                n1, n2 = choice(allChoice), choice(allChoice)
                if allChoice.index(n1) < allChoice.index(n2):
                    neuron1, neuron2 = n1, n2
                else:
                    neuron1, neuron2 = n2, n1

                self.connect(neuron1, neuron2)


    def clearNetwork(self):
        # Resets the network to blank state
        self.layers = []
        self.inputs = []
        self.outputs = []
        self.synapses = []

    def maxConnections(self):
        # Determines howmany synapses will be used for each layer to be fully connected to adjacent layers
        total = len(self.inputs)*len(self.layers[0])
        if len(self.layers) > 1:
            for i in range(0, len(self.layers)-1):
                total += len(self.layers[i])*len(self.layers[i+1])
        total += len(self.layers[-1])*len(self.outputs)
        return total

    def connect(self, neuron1, neuron2):

        connected = False
        for syn in neuron2.inputSynapses:
            if syn.neuronIn == neuron1:
                connected = True
                return False

        if not connected:
            self.addSynapse(neuron1, neuron2)
            return True

    def runInputs(self, inputs):

        for i in range(0, len(self.inputs)):
            self.inputs[i].value(inputs[i])
        self.step()
        out = []
        for i in self.outputs:
            out.append(i.value)
        return out

    def mutate(self):

        for syn in self.synapses:
            delta = abs(self.fitness - self.prevFitness)
            if delta > 10:
                syn.weight = gauss(syn.weight, SYNAPSE_SD/10)
            elif delta > 1:
                syn.weight = gauss(syn.weight, SYNAPSE_SD)
            else:
                syn.weight = gauss(syn.weight, SYNAPSE_SD*10)
        for layer in self.layers:
            for neuron in layer:
                neuron.B = abs(gauss(neuron.B, NEURON_B_SD))
                neuron.Q = abs(gauss(neuron.Q, NEURON_Q_SD))

        if uniform(0,1) < NEW_NEURON_RATE:
            a = choice([0,1])
            if a == 0:
                # Add neuron
                index = choice(list(range(0, len(self.layers)))+[None])
                self.addNeuron(index)

            else:
                # Remove neuron
                flatlist = self.neuronList()
                if len(flatlist):
                    neuron = choice(flatlist)
                    for syn in neuron.inputSynapses:
                        self.synapses.remove(syn)
                    for layer in self.layers:
                        try:
                            layer.remove(neuron)
                            if len(layer) == 0:
                                self.layers.remove(layer)
                            break
                        except:
                            pass

        if uniform(0,1) < NEW_SYNAPSE_RATE:
            a = choice([0,1])
            if a == 0:
                #Add synapse
                flatlist = self.neuronList(outputs = True)
                if len(flatlist) > len(self.outputs):
                    n1 = choice(self.inputs + flatlist)
                    try:
                        n2 = choice(flatlist[flatlist.index(n1)+1:])
                        tries = 1
                        while not self.connect(n1, n2):
                            if tries > len(flatlist):
                                break
                            n2 = choice(flatlist[flatlist.index(n1)+1:])
                            tries += 1
                    except:
                        pass


            else:
                # Remove synapse
                flatlist = self.neuronList()
                if len(flatlist):
                    neuron = choice(flatlist)
                    try:
                        syn = choice(neuron.inputSynapses)
                        neuron.inputSynapses.remove(syn)
                        self.synapses.remove(syn)
                    except:
                        pass

        if uniform(0,1)*2 < NEURON_SWAP_RATE:
            neuronType = choice(["Logistic", "Sum"])
            flatlist = []
            for layer in self.layers:
                    for neuron in layer:
                        flatlist.append(neuron)
            for out in self.outputs:
                flatlist.append(out)
            neuron = choice(flatlist)
            neuron.type = neuronType
            

    def countNeurons(self):

        self.neuronCount = 0
        for layer in self.layers:
            self.neuronCount += len(layer)

    def countSynapses(self):

        self.synapseCount = len(self.synapses)

    def neuronList(self, outputs = False):
        
        flatlist = []
        for layer in self.layers:
            for neuron in layer:
                flatlist.append(neuron)
        if outputs:
            for output in self.outputs:
                flatlist.append(output)
        return flatlist

    def copyNetwork(self):

        # Will return a new network that has identical properties to self
        newNet = Network()
        newInp = len(self.inputs)
        newOut = len(self.outputs)
        # Add the input and output layers
        newNet.addInputs(newInp)
        newNet.addOutputs(newOut)
        # Copy all the neurons over
        index1 = 0
        for layer in self.layers:
            newNet.addLayer(len(layer))
            index2 = 0
            for neuron in layer:
                newNet.layers[index1][index2].B = neuron.B
                newNet.layers[index1][index2].Q = neuron.Q
                newNet.layers[index1][index2].type = neuron.type
                index2 += 1
            index1 += 1
        # Copy all the synapses over
        for syn in self.synapses:
            # Python magic to find the layer index for input neuron
            # Index tells the location in self of the input neuron
            inputFlag = False
            outputFlag = False
            layerIndexIn = next((i for i, sublist in
                                 enumerate(self.layers)
                                 if syn.neuronIn in sublist),-1)
            if layerIndexIn == -1:
                inputFlag = True
                try:
                    layerIndexIn = self.inputs.index(syn.neuronIn)
                except:
                    '''print("Couldn't copy synapse")
                    print(self.inputs)
                    print(self.layers)
                    print(self.outputs)
                    print(syn.neuronIn)
                    print("")
                    self.countNeurons()
                    print("{} inputs nodes, {} neurons and {} output nodes, with {} synapses.".format(
                        len(self.inputs),
                        self.neuronCount,
                        len(self.outputs),
                        len(self.synapses)))'''
                    break
            if not inputFlag:
                otherIndexIn = self.layers[layerIndexIn].index(syn.neuronIn)
            # Find the index of the output neuron of that synapse
            layerIndexOut = -1
            for i in range(0,len(self.layers)):
                for j in range(0,len(self.layers[i])):
                    if syn in self.layers[i][j].inputSynapses:
                        layerIndexOut, otherIndexOut = i, j
                        break
                else:
                    continue
                break
            if layerIndexOut == -1:
                outputFlag = True
                for output in self.outputs:
                    if syn in output.inputSynapses:
                        layerIndexOut = self.outputs.index(output)
                        break
            if inputFlag and outputFlag:
                newNet.addSynapse(newNet.inputs[layerIndexIn], newNet.outputs[layerIndexOut])
            elif inputFlag and not outputFlag:
                newNet.addSynapse(newNet.inputs[layerIndexIn], newNet.layers[layerIndexOut][otherIndexOut])
            elif not inputFlag and outputFlag:
                newNet.addSynapse(newNet.layers[layerIndexIn][otherIndexIn], newNet.outputs[layerIndexOut])
            else:
                newNet.addSynapse(newNet.layers[layerIndexIn][otherIndexIn], newNet.layers[layerIndexOut][otherIndexOut])
            newNet.synapses[-1].weight = syn.weight

        return newNet
            

class Neuron():

    def __init__(self, Type = "Logistic"):

        self.B = NEURON_B_MEAN
        self.Q = NEURON_Q_MEAN
        self.output = 0
        self.inputSynapses = []
        self.type = Type

    def evaluate(self, n):

        if self.type == "Logistic":
            try:
                self.output = 1/(1 + self.Q*exp(-self.B*n))
            except:
                self.output = 0

        elif self.type == "Sum":
            self.output = n

        else:
            self.output = 0

    def getInputs(self):

        inp = 0
        for synapse in self.inputSynapses:
            inp += synapse.fire()
        #print("neuron total input is " + str(inp))
        self.evaluate(inp)
        #print("neuron output is " + str(self.output))

class Synapse():

    def __init__(self, neuronIn):

        self.weight = gauss(0, SYNAPSE_SD)
        self.output = 0
        self.neuronIn = neuronIn

    def fire(self):

        self.output = self.weight*self.neuronIn.output
        if self.output > MAX_SYNAPSE_OUTPUT:
            self.output = MAX_SYNAPSE_OUTPUT
        elif self.output < MIN_SYNAPSE_OUTPUT:
            self.output = MIN_SYNAPSE_OUTPUT
        #print("synapse input is " + str(self.neuronIn.output) + ", the output is " + str(self.output))
        return self.output

class InputNode():

    def __init__(self):
        self.output = 0

    def value(self, n):
        self.output = n
        
    
class OutputNode(Neuron):

    def __init__(self):
        Neuron.__init__(self, Type = "Logistic")
        self.value = 0
        self.inputSynapses = []

    def getInputs(self):

        inp = 0
        for synapse in self.inputSynapses:
            inp += synapse.fire()
        self.evaluate(inp)
        self.value = self.output

class Trainer():

    def __init__(self, netTemplate, networks, fitness, inputs, expectedOutputs, varyInputFunction = None, timing = False):

        # netTemplate is the network to seed the networks from
        # firstWarning is used if an invalid selection method is called
        self.netTemplate = netTemplate
        self.networks = []
        self.fitnessFunc = fitness
        self.inputFunc = varyInputFunction
        self.maxFitness = 0
        self.inputs = inputs
        self.expectedOutputs = expectedOutputs
        self.generations = 1
        self.firstWarning = True
        self.prevMaxFitness = []

        # timing is used to check the run time and the percent time used copying
        self.timing = timing
        if self.timing:
            self.timeStart = 0
            self.timeEnd = 0
            self.copyTime = 0

        # Generates the list of networks to be trained. One unaltered copy of the original is used
        self.networks.append(self.netTemplate.copyNetwork())
        self.networks[-1].fitness = 0
        for i in range(1, networks):
            #self.networks.append(deepcopy(self.netTemplate))
            self.networks.append(self.netTemplate.copyNetwork())
            self.networks[-1].mutate()
            self.networks[-1].fitness = 0

    def run(self, fitnessThreshold = 0.99, minGenerations = 100, selectionMethod = "Top10"):

        print("Running initialised")
        
        if self.timing:
            self.timeStart = clock()

        while (self.maxFitness < fitnessThreshold or self.generations < minGenerations):
            # Runs all the inputs through the networks and compares them to the expected output using the fitness function
            for net in self.networks:
                out = []
                for inp in self.inputs:
                    out.append(net.runInputs(inp))
                net.prevFitness = net.fitness
                net.fitness = self.fitnessFunc(out, self.expectedOutputs)
            # Sorts the networks from best to worse fitness
            self.networks = sorted(self.networks, key = lambda network: network.fitness)
            self.networks = list(reversed(self.networks))

            self.prevMaxFitness.append(self.maxFitness)
            while len(self.prevMaxFitness) > FITNESSS_HISTORY_LEN:
                del self.prevMaxFitness[0]

            averageFitness = sum(self.prevMaxFitness)/FITNESS_HISTORY_LEN
                
            self.maxFitness = self.networks[0].fitness

            if self.generations%50 == 0:
                self.Print()
                #pass

            # Selects the networks to be copied and mutates them
            self.select(selectionMethod)
            self.mutate()

            # Vary the inputs using some algorithm
            if self.inputFunc:
                inputs, outputs = self.inputFunc()

            self.generations += 1

        if self.timing:
            self.timeEnd = clock()
            self.totalTime = self.timeEnd - self.timeStart
            print("Time taken was {} seconds, with {}% time on copying".format(self.totalTime,
                                                                               100*self.copyTime/self.totalTime))
        return self.networks[0]
                    
                    

    def mutate(self):

        # Mutates all but the best network
        for net in self.networks[1:]:
            net.mutate()

    def select(self, method = "Top10"):

        if method == "Dynamic":
            # Alters the selection method to account for stagnation in the max fitness.
            

        elif method == "Top10":
            self.selectTop10()

        elif method == "Best1":
            self.selectBest1()

        elif method == "KeepAll":

            pass

        else:
            
            if self.firstWarning:
                print("Warning: '{}' is not a valid selection method.\nUsing default 'Top 10' method".format(method))
                self.firstWarning = False
            self.selectTop10()

    def selectTop10(self):

        # Takes the best 10% of networks and replaces the lower 90% in equal proportion
        new_gen_seed = self.networks[0:int(len(self.networks)/10)]
        self.networks = []
        for net in new_gen_seed:
            for i in range(0,10):
                if self.timing:
                    copyStart = clock()
                #self.networks.append(deepcopy(net))
                self.networks.append(net.copyNetwork())
                if self.timing:
                    copyEnd = clock()
                    self.copyTime += copyEnd - copyStart

    def selectBest1(self):

        # Takes the best network and replaces all others
        newGenSeed = self.networks[0]
        for i in range(0, len(self.networks)):
            if self.timing:
                copyStart = clock()
            self.networks[i] = newGenSeed.copyNetwork()
            if self.timing:
                copyEnd = clock()
                self.copyTime += copyEnd - copyStart


    def Print(self):

        print("Generation: {}".format(self.generations))
        print("Max fitness: {}\n".format(self.maxFitness))
        

if __name__ == '__main__':

    def f(a,b):
        dif = b[0] - a[0][0]
        return 1 - dif**2

    temp = Network()
    temp.addInputs(2)
    temp.addOutputs(1)
    temp.connect(temp.inputs[0], temp.outputs[0])
    temp.connect(temp.inputs[1], temp.outputs[0])

    test = Trainer(temp, 100, f, [[0.1,0.2]], [0.3], timing = True)
    a = test.run(0.9999, 200)

''' Changing the deepcopy call to the custom copy function in the network class
    reduced the run time by ~80%, reducing the time spent copying from 90% of
    the run time to 60%.
'''
    
