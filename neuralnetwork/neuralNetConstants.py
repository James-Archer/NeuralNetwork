# CONSTANTS FOR THE NEURAL NETWORK. COPY THIS FILE AND IMPORT
# FROM THE NEW FILE TO CHANGE VALUES

# FOR THE NETWORK

SYNAPSE_SD = 0.1
SYNAPSE_SD_MAX = 0.1
NEURON_B_MEAN = 1
NEURON_B_SD = 1
NEURON_Q_MEAN = 1
NEURON_Q_SD = 1

NEW_NEURON_RATE = 0.01
NEW_SYNAPSE_RATE = 0.1
NEURON_SWAP_RATE = 0.01

MAX_SYNAPSE_OUTPUT = 30
MIN_SYNAPSE_OUTPUT = -30

# FOR THE TRAINER

DYNAMIC_THRESHOLD = 0.0001
FITNESS_HISTORY_LEN = 5
MAX_STAGNANT_GENERATIONS = 200
STAGNANT_MUTATE_WEIGHT = 10
MUTATE_RATE = 5
MAX_MUTATE_GROWTH = 10
FITNESS_PEAK_THRESHOLD = 0.1
ALL_IN_FACTOR = 100
ALL_IN_NUMBER = 100
