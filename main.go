package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

// Constants for the simulation
const (
	ScreenWidth    = 800
	ScreenHeight   = 600
	PopulationSize = 50
	TargetRadius   = 20
	AgentRadius    = 10
	MaxSpeed       = 2.0
)

// Agent represents an individual in the simulation
type Agent struct {
	position rl.Vector2
	velocity rl.Vector2
	network  *NeuralNetwork
}

// Target represents the goal the agents need to follow
type Target struct {
	position rl.Vector2
}

// CreateAgents initializes the agents with random positions and velocities
func CreateAgents(ga *GeneticAlgorithm) []*Agent {
	agents := make([]*Agent, ga.populationSize)
	for i := 0; i < ga.populationSize; i++ {
		agents[i] = &Agent{
			position: rl.Vector2{X: rand.Float32() * ScreenWidth, Y: rand.Float32() * ScreenHeight},
			velocity: rl.Vector2{X: rand.Float32()*2 - 1, Y: rand.Float32()*2 - 1},
			network:  ga.population[i],
		}
	}
	return agents
}

// FitnessFunction calculates how close an agent is to the target
func FitnessFunction(agent *Agent, target *Target) float64 {
	distance := math.Sqrt(math.Pow(float64(agent.position.X-target.position.X), 2) + math.Pow(float64(agent.position.Y-target.position.Y), 2))
	return 1.0 / (distance + 1.0) // Higher fitness for being closer
}

// UpdateAgent moves the agent based on its neural network's output
func UpdateAgent(agent *Agent, target *Target) {
	// Inputs: normalized distance and direction to the target
	dx := target.position.X - agent.position.X
	dy := target.position.Y - agent.position.Y
	distance := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	inputs := []float64{
		float64(dx) / float64(ScreenWidth),
		float64(dy) / float64(ScreenHeight),
		float64(distance) / float64(math.Max(ScreenWidth, ScreenHeight)),
	}

	outputs := agent.network.Forward(inputs)
	agent.velocity.X = float32(outputs[0])*2 - 1 // Convert output to range [-1, 1]
	agent.velocity.Y = float32(outputs[1])*2 - 1

	// Limit speed
	speed := float32(math.Sqrt(float64(agent.velocity.X*agent.velocity.X + agent.velocity.Y*agent.velocity.Y)))
	if speed > MaxSpeed {
		agent.velocity.X *= MaxSpeed / speed
		agent.velocity.Y *= MaxSpeed / speed
	}

	// Update position
	agent.position.X += agent.velocity.X
	agent.position.Y += agent.velocity.Y

	// Keep within screen bounds
	if agent.position.X < 0 {
		agent.position.X = 0
	}
	if agent.position.X > ScreenWidth {
		agent.position.X = ScreenWidth
	}
	if agent.position.Y < 0 {
		agent.position.Y = 0
	}
	if agent.position.Y > ScreenHeight {
		agent.position.Y = ScreenHeight
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Initialize Raylib
	rl.InitWindow(ScreenWidth, ScreenHeight, "Agent Target Training")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	// Create target and genetic algorithm
	target := &Target{
		position: rl.Vector2{X: rand.Float32() * ScreenWidth, Y: rand.Float32() * ScreenHeight},
	}
	ga := NewGeneticAlgorithm(3, 2, PopulationSize)
	agents := CreateAgents(ga)

	// Training loop
	generation := 0
	for !rl.WindowShouldClose() {
		// Draw the simulation
		rl.BeginDrawing()
		rl.ClearBackground(rl.RayWhite)

		// Draw target
		rl.DrawCircleV(target.position, TargetRadius, rl.Red)

		// Update agents and draw them
		fitnesses := make([]float64, len(agents))
		for i, agent := range agents {
			UpdateAgent(agent, target)
			rl.DrawCircleV(agent.position, AgentRadius, rl.Blue)

			// Calculate fitness
			fitnesses[i] = FitnessFunction(agent, target)
		}

		// Update target position randomly every 300 frames
		if rl.GetFrameTime() > 300 {
			target.position = rl.Vector2{X: rand.Float32() * ScreenWidth, Y: rand.Float32() * ScreenHeight}
		}

		// Evolve the population every 300 frames
		if generation%300 == 0 {
			ga.Evolve(func(nn *NeuralNetwork) float64 {
				for i, agent := range agents {
					if agent.network == nn {
						return fitnesses[i]
					}
				}
				return 0.0
			})

			agents = CreateAgents(ga)
			generation++
		}

		rl.DrawText(fmt.Sprintf("Generation: %d", generation), 10, 10, 20, rl.Black)
		rl.EndDrawing()
	}
}
