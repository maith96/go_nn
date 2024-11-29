package main

import (
	"math"
	"math/rand"
)

// Neuron represents a single neuron in the network
type Neuron struct {
	id          int
	layer       int
	connections map[int]*Connection
	value       float64
	activations float64 // Track how often neuron fires
	lastActive  float64 // Last activation value
}

// Connection represents a synaptic connection between neurons
type Connection struct {
	weight   float64
	sourceID int
	targetID int
	strength float64 // How often this connection is used
	lastUsed float64 // When was this connection last used
}

type NeuralNetwork struct {
	neurons             map[int]*Neuron
	layers              [][]int // Store neuron IDs by layer
	inputCount          int
	outputCount         int
	nextNeuronID        int
	activationThreshold float64
	growthThreshold     float64
	pruneThreshold      float64
}

func NewNeuralNetwork(inputCount, outputCount int) *NeuralNetwork {
	nn := &NeuralNetwork{
		neurons:             make(map[int]*Neuron),
		layers:              make([][]int, 3), // Start with input, hidden, output layers
		inputCount:          inputCount,
		outputCount:         outputCount,
		nextNeuronID:        0,
		activationThreshold: 0.5,
		growthThreshold:     0.8,
		pruneThreshold:      0.2,
	}

	// Create input layer
	for i := 0; i < inputCount; i++ {
		nn.addNeuron(0) // Layer 0 is input
	}

	// Create initial hidden layer with 2x input size
	hiddenCount := inputCount * 2
	for i := 0; i < hiddenCount; i++ {
		neuronID := nn.addNeuron(1)
		// Connect to all input neurons
		for inputID := range nn.layers[0] {
			nn.addConnection(inputID, neuronID)
		}
	}

	// Create output layer
	for i := 0; i < outputCount; i++ {
		neuronID := nn.addNeuron(2)
		// Connect to all hidden neurons
		for hiddenID := range nn.layers[1] {
			nn.addConnection(hiddenID, neuronID)
		}
	}

	return nn
}

func (nn *NeuralNetwork) addNeuron(layer int) int {
	neuron := &Neuron{
		id:          nn.nextNeuronID,
		layer:       layer,
		connections: make(map[int]*Connection),
	}
	nn.neurons[neuron.id] = neuron
	nn.layers[layer] = append(nn.layers[layer], neuron.id)
	nn.nextNeuronID++
	return neuron.id
}

func (nn *NeuralNetwork) addConnection(sourceID, targetID int) {
	source := nn.neurons[sourceID]
	target := nn.neurons[targetID]

	if source.layer >= target.layer {
		return // Prevent backwards/same layer connections
	}

	connection := &Connection{
		weight:   rand.Float64()*2 - 1,
		sourceID: sourceID,
		targetID: targetID,
		strength: 1.0,
	}

	target.connections[sourceID] = connection
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (nn *NeuralNetwork) Forward(inputs []float64) []float64 {
	// Set input values
	for i, input := range inputs {
		if i < len(nn.layers[0]) {
			nn.neurons[nn.layers[0][i]].value = input
		}
	}

	// Process hidden and output layers
	for layer := 1; layer < len(nn.layers); layer++ {
		for _, neuronID := range nn.layers[layer] {
			neuron := nn.neurons[neuronID]
			sum := 0.0

			for _, conn := range neuron.connections {
				source := nn.neurons[conn.sourceID]
				sum += source.value * conn.weight
				conn.lastUsed = conn.strength
			}

			neuron.lastActive = neuron.value
			neuron.value = sigmoid(sum)

			// Update neuron activation history
			if neuron.value > nn.activationThreshold {
				neuron.activations += 0.1

				// Strengthen frequently used connections
				for _, conn := range neuron.connections {
					conn.strength += 0.1 * neuron.value
				}
			}
		}
	}

	// Growth and pruning phase
	nn.evolveNetwork()

	// Collect outputs
	outputs := make([]float64, nn.outputCount)
	for i, neuronID := range nn.layers[len(nn.layers)-1] {
		if i < len(outputs) {
			outputs[i] = nn.neurons[neuronID].value
		}
	}

	return outputs
}

func (nn *NeuralNetwork) evolveNetwork() {
	// Add new neurons if existing ones are very active
	for _, neuronID := range nn.layers[1] { // Check hidden layer
		neuron := nn.neurons[neuronID]
		if neuron.activations > nn.growthThreshold {
			// Add a new neuron
			newID := nn.addNeuron(1)

			// Connect it to active inputs
			for _, inputID := range nn.layers[0] {
				if nn.neurons[inputID].value > nn.activationThreshold {
					nn.addConnection(inputID, newID)
				}
			}

			// Connect to outputs
			for _, outputID := range nn.layers[2] {
				nn.addConnection(newID, outputID)
			}

			neuron.activations = 0 // Reset activation counter
		}
	}

	// Prune weak connections
	for _, neuron := range nn.neurons {
		for sourceID, conn := range neuron.connections {
			if conn.strength < nn.pruneThreshold {
				delete(neuron.connections, sourceID)
			}
		}
	}
}

// GetWeights returns a flattened array of all weights for genetic algorithm
func (nn *NeuralNetwork) GetWeights() []float64 {
	weights := []float64{}
	for _, neuron := range nn.neurons {
		for _, conn := range neuron.connections {
			weights = append(weights, conn.weight)
		}
	}
	return weights
}

// SetWeights updates network weights from a flattened array
func (nn *NeuralNetwork) SetWeights(weights []float64) {
	index := 0
	for _, neuron := range nn.neurons {
		for _, conn := range neuron.connections {
			if index < len(weights) {
				conn.weight = weights[index]
				index++
			}
		}
	}
}
