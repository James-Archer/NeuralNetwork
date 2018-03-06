from math import exp
from random import gauss, gammavariate, choice, uniform
#from copy import deepcopy
from time import clock
from .neuralNetConstants import *  
  

class Network():

    def __init__(self):

        self.layers = []
        self.inputs = []
        self.outputs = []
        self.synapses = []
        self.fitness = 0
        self.prevFitness = 0
        
    def Print(self):
        for inp in self.inputs:
            print(inp)
        for layer in self.layers:
            print(layer)
            for neuron in layer:
                print(neuron)
        for out in self.outputs:
            print(out)

    def addInputs(self, n):
        # Adds n input nodes
        temp = []
        for i in range(0,n):
            temp.append(InputNode())
            temp[-1].neuronIndex = i
            temp[-1].layerIndex = "Inputs"
        self.inputs = temp

    def addOutputs(self, n):
        # Adds n output nodes
        temp = []
        for i in range(0,n):
            temp.append(OutputNode())
            temp[-1].neuronIndex = i
            temp[-1].layerIndex = "Outputs"
        self.outputs = temp

    def addLayer(self, n, index = None):
        # Adds a layer of n nodes. If no second arg is specified, goes to end
        layer = Layer()
        for i in range(0,n):
            layer.addNeuron(Neuron())
        if index:
            self.layers.insert(index, layer)
            for lay in self.layers:
                lay.layerIndex = self.layers.index(lay)
                for neuron in lay:
                    neuron.layerIndex = lay.layerIndex
        else:
            self.layers.append(layer)
            self.layers[-1].layerIndex = self.layers.index(layer)
            for neuron in self.layers[-1]:
                neuron.layerIndex = self.layers[-1].layerIndex

    def addSynapse(self, neuronIn, neuronOut):

        # Connects two neurons (or inputs/outputs) with a synapse
        self.synapses.append(Synapse(neuronIn, neuronOut))
        neuronOut.inputSynapses.append(self.synapses[-1])

    def addNeuron(self, layerIndex = None):
        # Inserts a neuron into a layer. If no index is specified, a new layer is added to end
        if layerIndex:
            self.layers[layerIndex].addNeuron(Neuron())
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

    def mutate(self, weight = 1):
        # Inceases from 1 to MAX_MUTATE_GROWTH as the weight increases
        chanceWeight = 1 + ((MAX_MUTATE_GROWTH - 1)/(MAX_STAGNANT_GENERATIONS*STAGNANT_MUTATE_WEIGHT - 1)**(1/MUTATE_RATE))*(weight - 1)**(1/MUTATE_RATE)

        for syn in self.synapses:
            syn.weight = gauss(syn.weight, SYNAPSE_SD*weight)
        for layer in self.layers:
            for neuron in layer:
                neuron.B = abs(gauss(neuron.B, NEURON_B_SD*weight))
                neuron.Q = abs(gauss(neuron.Q, NEURON_Q_SD*weight))

        if uniform(0,1) < NEW_NEURON_RATE*chanceWeight:
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

        if uniform(0,1) < NEW_SYNAPSE_RATE*chanceWeight:
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

        if uniform(0,1) < NEURON_SWAP_RATE*chanceWeight:
            neuronType = choice(["Logistic", "Sum"])
            flatlist = self.neuronList(outputs = True)
            neuron = choice(flatlist)
            neuron.type = neuronType


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
        # Add the input and output layers
        newNet.addInputs(len(self.inputs))
        newNet.addOutputs(len(self.outputs))
        # Copy all the layers, neuron copy implicit
        for layer in self.layers:
            # Fix layer index errors in just one line. Bugs hate it!
            layer.updateIndexes()
            newNet.layers.append(layer.copy())
        # Copy all the synapses over
        for syn in self.synapses:
            try:
                if syn.neuronIn.layerIndex == "Inputs":
                    if syn.neuronOut.layerIndex == "Outputs":
                            newNet.connect(newNet.inputs[syn.neuronIn.neuronIndex],
                                       newNet.outputs[syn.neuronOut.neuronIndex])
                    else:
                            newNet.connect(newNet.inputs[syn.neuronIn.neuronIndex],
                                       newNet.layers[syn.neuronOut.layerIndex][syn.neuronOut.neuronIndex])
                elif syn.neuronOut.layerIndex == "Outputs":
                    newNet.connect(newNet.layers[syn.neuronIn.layerIndex][syn.neuronIn.neuronIndex],
                                   newNet.outputs[syn.neuronOut.neuronIndex])
                else:
                    newNet.connect(newNet.layers[syn.neuronIn.layerIndex][syn.neuronIn.neuronIndex],
                                   newNet.layers[syn.neuronOut.layerIndex][syn.neuronOut.neuronIndex])
                newNet.synapses[-1].weight = syn.weight
            except:
                #self.debug(syn)
                pass

        return newNet

    def debug(self, problem):
        if type(problem) == Synapse:
        # Exists purely to characterise why the layer index was fucked up. Keeping in case of emergencies...
            print("Problem. Synapse details are:")
            print("Input neuron: {}\n\tLayer Index: {}\n\tNeuron Index: {}".format(problem.neuronIn,
                                                                                  problem.neuronIn.layerIndex,
                                                                                  problem.neuronIn.neuronIndex))
            print("Output neuron: {}\n\tLayer Index: {}\n\tNeuron Index: {}".format(problem.neuronOut,
                                                                                  problem.neuronOut.layerIndex,
                                                                                  problem.neuronOut.neuronIndex))
        for lay in self.layers:
            print("Layer {}".format(lay.layerIndex))
            for neu in lay:
                print("\t{}".format(neu))


class Neuron():

    def __init__(self, Type = "Logistic"):

        self.B = NEURON_B_MEAN
        self.Q = NEURON_Q_MEAN
        self.output = 0
        self.inputSynapses = []
        self.type = Type
        self.neuronIndex = None
        self.layerIndex = None
        
    def __str__(self):
        
        return f"Neuron: {self.B}, {self.Q}"

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

    def copy(self):
        newNeuron = Neuron()
        newNeuron.B = self.B
        newNeuron.Q = self.Q
        newNeuron.type = self.type
        newNeuron.neuronIndex = self.neuronIndex
        return newNeuron

class Synapse():

    def __init__(self, neuronIn, neuronOut):

        self.weight = gauss(0, SYNAPSE_SD)
        self.output = 0
        self.neuronIn = neuronIn
        self.neuronOut = neuronOut
        
    def __str__(self):
        
        return f"Synapse: {self.weight}"

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
        self.neuronIndex = None
        
    def __str__(self):
        
        return f"Input: {self.neuronIndex}"

    def value(self, n):
        self.output = n
        
    
class OutputNode(Neuron):

    def __init__(self):
        Neuron.__init__(self, Type = "Logistic")
        self.value = 0
        self.inputSynapses = []
        
    def __str__(self):
        
        return f"Output: {self.neuronIndex}"

    def getInputs(self):

        inp = 0
        for synapse in self.inputSynapses:
            inp += synapse.fire()
        self.evaluate(inp)
        self.value = self.output


class Layer(list):

    def __init__(self):

        self.layerIndex = None
        
    def __str__(self):
        
        return f"Layer: {self.layerIndex}"

    def addNeuron(self, neuron):

        neuron.layerIndex = self.layerIndex
        self.append(neuron)
        self[-1].neuronIndex = self.index(neuron)

    def updateIndexes(self):

        for i in range(0,len(self)):
            self[i].neuronIndex = i
            self[i].layerIndex = self.layerIndex

    def copy(self):

        newLayer = Layer()
        for neuron in self:
            newLayer.addNeuron(neuron.copy())
        newLayer.layerIndex = self.layerIndex
        return newLayer

class Trainer():

    def __init__(self, netTemplate, networks, fitness, inputs, expectedOutputs, varyInputFunction = None, timing = False):

        # netTemplate is the network to seed the networks from
        # firstWarning is used if an invalid selection method is called
        self.netTemplate = netTemplate
        self.networks = []
        self.fitnessFunc = fitness
        self.inputFunc = varyInputFunction
        self.maxFitness = 0
        self.averageFitness = -999
        self.selectionMethod = "Top10"
        self.inputs = inputs
        self.expectedOutputs = expectedOutputs
        self.generations = 1
        self.firstWarning = True
        self.prevMaxFitness = []
        self.bestLocalFitness = None
        self.stagnantFlag = False
        self.deadEndFlag = False
        self.allInFlag = False

        # timing is used to check the run time and the percent time used copying
        self.timing = timing
        if self.timing:
            self.timeStart = 0
            self.timeEnd = 0
            self.copyTime = 0
            self.mutateTime = 0

        # Generates the list of networks to be trained. One unaltered copy of the original is used
        self.networks.append(self.netTemplate.copyNetwork())
        self.networks[-1].fitness = 0
        for i in range(1, networks):
            #self.networks.append(deepcopy(self.netTemplate))
            self.networks.append(self.netTemplate.copyNetwork())
            self.networks[-1].mutate()
            self.networks[-1].fitness = 0

    def step(self):

            # Selects the networks to be copied and mutates them
            self.select(self.selectionMethod)
            if not self.stagnantFlag:
                self.mutate()
            
            # Run the networks to find the fitness
            for net in self.networks:
                out = []
                for inp in self.inputs:
                    out.append(net.runInputs(inp))
                net.prevFitness = net.fitness
                net.fitness = self.fitnessFunc(out, self.expectedOutputs)
            # Sorts the networks from best to worse fitness
            self.networks = sorted(self.networks, key = lambda network: network.fitness)
            self.networks = list(reversed(self.networks))

            # Keeps track of the mean best fitness
            self.prevMaxFitness.append(self.maxFitness)
            while len(self.prevMaxFitness) > FITNESS_HISTORY_LEN:
                del self.prevMaxFitness[0]

            self.averageFitness = sum(self.prevMaxFitness)/FITNESS_HISTORY_LEN
                
            self.maxFitness = self.networks[0].fitness


    def run(self, fitnessThreshold = 0.99, minGenerations = 100, selectionMethod = "Top10"):

        print("Running initialised")

        self.selectionMethod = selectionMethod
        if self.timing:
            self.timeStart = clock()

        while ((self.maxFitness < fitnessThreshold or self.generations < minGenerations) and not self.deadEndFlag):
            # Runs all the inputs through the networks and compares them to the expected output using the fitness function
            self.step()

            if self.timing:
                        self.timeEnd = clock()
                        self.totalTime = self.timeEnd - self.timeStart
                        #print("Time taken was {} seconds, with {}% time on copying and {}% on mutating.".format(self.totalTime,
                        #                                                                   100*self.copyTime/self.totalTime,
                        #                                                                   100*self.mutateTime/self.totalTime))

            if self.generations%50 == 0:
                self.Print()

            # Vary the inputs using some algorithm
            if self.inputFunc:
                self.inputs, self.expectedOutputs = self.inputFunc()

            if self.deadEndFlag:
                print("Trainer ran into a local maximum in fitness.")
                
        if self.timing:
            print("Time taken was {} seconds, with {}% time on copying and {}% on mutating.".format(self.totalTime,
                                                                                           100*self.copyTime/self.totalTime,
                                                                                           100*self.mutateTime/self.totalTime))
            
        return self.networks[0]
                    
                    

    def mutate(self, n = 1, weight = 1):
        if self.timing:
            self.mutateTime -= clock()
        # Mutates all but the best network
        for i in range(0,n):
            for net in self.networks[1:]:
                net.mutate(weight)
        self.generations += n
        if self.timing:
            self.mutateTime += clock()

    def select(self, method = "Top10"):

        if method == "Dynamic":
            # Alters the selection method to account for stagnation in the max fitness.
            if not self.stagnantFlag:
                # If not stagnant, check to see if it is. If so, flag True and save the current best
                if (self.maxFitness - self.averageFitness < DYNAMIC_THRESHOLD) and (self.maxFitness < 1 - FITNESS_PEAK_THRESHOLD):
                    self.lastBestFitness = self.maxFitness
                    self.stagnantFlag = True
                    self.stagnantGenerations = 0
                    print("Stagnated...")
                # If not stagnent, do top 10 selection
                else:
                    self.selectTop10()
            # If stagnation has occured, allow random walks with zero pressure
            else:
                self.stagnantGenerations += 1
                self.mutate(weight = self.stagnantGenerations*STAGNANT_MUTATE_WEIGHT*(1+ALL_IN_FACTOR*self.allInFlag),
                            n = 1 + self.allInFlag*ALL_IN_NUMBER)
                if self.allInFlag:
                    self.deadEndFlag = True
                if self.stagnantGenerations > MAX_STAGNANT_GENERATIONS:
                    if not self.allInFlag:
                        print("Fuck it, {} rounds of going crazy".format(ALL_IN_NUMBER))
                        self.allInFlag = True
                # Check to see if the random walks have improved upon the last best value
                if self.maxFitness > self.lastBestFitness:
                    self.stagnantFlag = False
                    print("Stagnation broken, continuing normally")
                    

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

    def vary():
        a = uniform(-10, 10)
        b = uniform(-10, 10)
        return [[a,b]], [a+b]
        
    temp = Network()
    temp.addInputs(2)
    temp.addOutputs(1)
    temp.outputs[0].type = "Sum"
    temp.connect(temp.inputs[0], temp.outputs[0])
    temp.connect(temp.inputs[1], temp.outputs[0])
    for syn in temp.synapses:
        syn.weight = 1

    test = Trainer(temp, 100, f, [[0.1,0.2], [3,2], [-0.5,0.5],[0.2,-0.6]], [0.3, 5, 0, -0.4], timing = True, varyInputFunction = None)
    a = test.run(0.9999, 200)

''' Changing the deepcopy call to the custom copy function in the network class
    reduced the run time by ~80%, reducing the time spent copying from 90% of
    the run time to 60%.
'''
    
