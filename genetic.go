package main

import (
	"math"
	"math/rand"
	"sort"
)

// GeneticAlgorithm represents the genetic algorithm for evolving neural networks
type GeneticAlgorithm struct {
	populationSize int
	mutationRate   float64
	crossoverRate  float64
	population     []*NeuralNetwork
}

// NewGeneticAlgorithm initializes a new genetic algorithm with a population of networks
func NewGeneticAlgorithm(inputCount, outputCount, populationSize int) *GeneticAlgorithm {
	ga := &GeneticAlgorithm{
		populationSize: populationSize,
		mutationRate:   0.05,
		crossoverRate:  0.7,
		population:     make([]*NeuralNetwork, populationSize),
	}

	for i := 0; i < populationSize; i++ {
		ga.population[i] = NewNeuralNetwork(inputCount, outputCount)
	}

	return ga
}

// EvaluateFitness evaluates each network's fitness using a fitness function
func (ga *GeneticAlgorithm) EvaluateFitness(fitnessFunc func(*NeuralNetwork) float64) []float64 {
	fitnesses := make([]float64, ga.populationSize)
	for i, nn := range ga.population {
		fitnesses[i] = fitnessFunc(nn)
	}
	return fitnesses
}

// SelectParents selects two parents using tournament selection
func (ga *GeneticAlgorithm) SelectParents(fitnesses []float64) (*NeuralNetwork, *NeuralNetwork) {
	tournamentSize := 3

	// Select first parent
	parent1 := ga.selectParentByTournament(fitnesses, tournamentSize)

	// Select second parent
	parent2 := ga.selectParentByTournament(fitnesses, tournamentSize)

	return parent1, parent2
}

func (ga *GeneticAlgorithm) selectParentByTournament(fitnesses []float64, tournamentSize int) *NeuralNetwork {
	bestIndex := -1
	bestFitness := math.Inf(-1)

	for i := 0; i < tournamentSize; i++ {
		index := rand.Intn(ga.populationSize)
		if fitnesses[index] > bestFitness {
			bestFitness = fitnesses[index]
			bestIndex = index
		}
	}

	return ga.population[bestIndex]
}

// Crossover performs crossover between two parent networks to create a child network
func (ga *GeneticAlgorithm) Crossover(parent1, parent2 *NeuralNetwork) *NeuralNetwork {
	child := NewNeuralNetwork(parent1.inputCount, parent1.outputCount)

	if rand.Float64() < ga.crossoverRate {
		weights1 := parent1.GetWeights()
		weights2 := parent2.GetWeights()

		crossoverPoint := rand.Intn(len(weights1))
		childWeights := append(weights1[:crossoverPoint], weights2[crossoverPoint:]...)
		child.SetWeights(childWeights)
	} else {
		// No crossover; clone one of the parents
		child.SetWeights(parent1.GetWeights())
	}

	return child
}

// Mutate applies mutation to the neural network's weights
func (ga *GeneticAlgorithm) Mutate(network *NeuralNetwork) {
	weights := network.GetWeights()
	for i := range weights {
		if rand.Float64() < ga.mutationRate {
			weights[i] += rand.NormFloat64() * 0.1 // Add small Gaussian noise
		}
	}
	network.SetWeights(weights)
}

// Evolve generates the next generation of neural networks
func (ga *GeneticAlgorithm) Evolve(fitnessFunc func(*NeuralNetwork) float64) {
	fitnesses := ga.EvaluateFitness(fitnessFunc)

	// Sort population by fitness
	sortedIndices := make([]int, ga.populationSize)
	for i := range sortedIndices {
		sortedIndices[i] = i
	}
	sort.Slice(sortedIndices, func(i, j int) bool {
		return fitnesses[sortedIndices[i]] > fitnesses[sortedIndices[j]]
	})

	// Keep top performers
	nextPopulation := make([]*NeuralNetwork, ga.populationSize)
	for i := 0; i < ga.populationSize/2; i++ {
		nextPopulation[i] = ga.population[sortedIndices[i]]
	}

	// Generate new offspring
	for i := ga.populationSize / 2; i < ga.populationSize; i++ {
		parent1, parent2 := ga.SelectParents(fitnesses)
		child := ga.Crossover(parent1, parent2)
		ga.Mutate(child)
		nextPopulation[i] = child
	}

	ga.population = nextPopulation
}
