# NeuralNetwork
A layered neural network. Designed by James Archer and [Matthew Griffiths](https://github.com/Sever0us). If you've stumbled across this for whatever reason, this is *NOT* the code you're looking for. Feel free to have a look but you'll probably just end up hating me and everything I've ever done. If you can use this in any project; firstly, I'm so sorry, and secondly, feel free to contribute.

Mostly developed for [this project](https://github.com/James-Archer/HungryPond). The `Trainer` class is not used in that and so is very unlikely to be further developed.

Classes included in this module:

Network - This is an individual network comprised of inputNodes, outputNodes and layers of Neurons, all connected by Synapses to perform non-linear calculations between inputNodes and outputNodes.

Neuron - The nodes where calculations are performed on one or more inputs, received from Synapses. Various operations can be performed on the inputs. 

Synapse - The links between Neurons and inputNodes/outputNodes. They take an input signal, multiply by a weight and sends to the outputs when called upon.

InputNode - The inputs where data is fed into the network. Is treated like a Neuron by Synapses.

OutputNode - Subclass of Neuron. Will store the outputs from the network.

Trainer - A collection of networks, inputs and expected outputs to train the networks on.


Network

Parameters:
* layers - a list of layers containing the Neurons. 
* inputs - a list of the inputNodes.
* outputs - a list of the outputNodes.
* synapses - a list of all Synapses. Some of the Synapses may not be linked to any Neurons, due to the mutate function.
* neuronCount - the number of Neurons in the Network.
* fitness - the current fitness of the Network, evaluated by an external fitness function, usually handled by the Trainer class.

Functions:
* addInputs(n) - adds n inputNodes to the inputs list.
* addOutputs(n) - adds n outputNodes to the outputs list. Default to the logistic function.
* addLayer(n, index = None) - adds a list to the layers list with n Neurons in this list. If index is specified it will insert the layer into that position, otherwise it will go to the end.
* addSynapse(neuronic, neuronOut) - creates a Synapse linking neuronic to neuronOut. Don’t use this to add a Synapse to the Network, use connect()
* addNeuron(layerIndex = None) - adds a Neuron to the Network, in layer layerIndex. If not specified, the new Neuron will go in a new layer at the end.
* step() - runs the inputs through the Network and assign the outputNodes.value
* createNetwork(inOut, layers) - clears the Network nodes and connections and creates a new one. inOut is a list of the form (number of inputNodes, number of outputNodes), and layers is a list of the number of Neurons to be in each layer. Does not add any Synapses.
* populateNetwork(synType = “Full”, clear = False, completeness = 0.5) - this populates the Network with Synapses. If synType = “Full” then for each layer, all the Neurons (or outputNodes) will have a synapse linking them to each Neuron (or inputNode) in the preceding layer. If synType = “Random”, then there will be a number of Synapses randomly linking Neurons. All input nodes and outputNodes are guaranteed to have at least one Synapse linked to them, and each Neuron will have at least one input and output Synapse. The total number of Synapses depends on the completeness argument.
* clearNetwork() - removes all Neurons and Synapses. Is called by createNetwork()
* maxConnections() - calculates the maximum number of Synapses the Network can have using the “Full” method in populateNetwork(). Returns this number.
* connect(neuron1, neuron2) - checks to see if the two Neurons have a Synapse between them already, and if not, will add one.
* runInputs(inputs) - takes a list of inputs and assigns them to Network.inputs. It then calls step() and returns a list of the outputs. 
* mutate() - Will add or remove Synapses, Neurons and alter the Neuron type randomly depending on the weights defined at the top of the file.
* countNeurons() - updates neuronCount.
* countSynapses() - updates synapseCount.

Neuron

Parameters:
* output - the value that is retrieved by a Synapse. Is updated by getInputs()
* inputSynapses - a list of the Synapses that link into the Neuron.
* type - the type of operation the Neuron performs. Can be “Logistic” (default) or “Sum”.
* B - the parameter used in the logistic function to determine the steepness.
* Q - the parameter used by the logistic function to determine where the step begins.

Functions:
* getInputs() - takes the output value from all the Synapses in inputSynapses and sums them up. Passes this sum into evaluate()
* evaluate(inp) - runs the inp through the operation assigned to the Neuron, then updates the output value to this.

Synapse

Parameters:
* weight - the weight that the input is multiplied by.
* output - the output value to be called by the Neuron linked to this Synapse.
* neuronIn - the Neuron that the Synapse takes the output of.

Functions:
* fire() - takes the input from the Neuron (or inputNode), and multiplies it by the Synapses weight. If the output is outside the range defined by MAX_SYNAPSE_OUTPUT or MIN_SYNAPSE_OUTPUT it is set to the appropriate max/min, then this value is set to the output of the Synapse.

InputNode

Parameters:
* output - the value retrieved by a Synapse.

Functions:
* value(n) - sets the output to n.

OutputNode

Parameters:
* value - the final output value for this node of the Network. Otherwise is identical to a Neuron.


Trainer

Parameters:
* netTemplate - a Network that will be used as the basis for all the Networks in the list.
* networks - a list of the Networks in the Trainer.
* fitnessFunc(outputs, expectedOutputs) - the function used to evaluate the fitness value of a Network. Must take arguments of a list of the Network outputs and the expected outputs. Returns a number, usually with a maximum of 1. 
* inputs - a list of the inputs to train the Networks with.
* expectedOutputs - a list of the expected outputs for the network.
* generations - the number of generation the Networks have gone through.

Functions:
* run(fitnessThreshold = 0.99, minGenerations = 100) - runs the Trainer until the best Network has a fitness greater than fitnessThreshold and the number of generations are greater than minGenerations.
* mutate() - mutates all the Networks.
* select(method = “Top10”) - Chooses which Networks to propagate into the next generation. “Top10” takes the top 10% of Networks and makes ten copies of each, mutating them. Keeps a copy of the best performing Network unchanged.
* Print() - prints off the generation number, and max fitness of the Networks.
