import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        # Initialize parameters
        self.parameters = {}
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            layer_in = layer_sizes[i-1]
            layer_out = layer_sizes[i]
            self.parameters[f"W{i}"] = np.random.randn(layer_in, layer_out)
            self.parameters[f"b{i}"] = np.random.randn(layer_out)
    

    def initialize_parameters(self):
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.parameters = {}
        for i in range(1, len(sizes)):
            self.parameters[f"W{i}"] = np.random.randn(sizes[i-1], sizes[i])
            self.parameters[f"b{i}"] = np.zeros((1, sizes[i]))

    def forward(self, X):
        self.activations = {"A0": X}

        for i in range(1, self.num_layers + 1):
            W = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]
            Z = np.dot(self.activations[f"A{i-1}"], W) + b
            A = self.activation_function(Z)
            self.activations[f"A{i}"] = A

        return self.activations[f"A{self.num_layers}"]
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def evolve(self, neural_network, X, y, num_generations):
        population = self.initialize_population(neural_network)
        for generation in range(num_generations):
            fitness_scores = self.calculate_fitness(neural_network, population, X, y)
            best_individual_idx = np.argmax(fitness_scores)
            best_individual = population[best_individual_idx]
            print("Generation:", generation)
            print("Best Individual Fitness:", fitness_scores[best_individual_idx])
            print("Best Individual Parameters:", best_individual.parameters)
            print("")
            population = self.crossover(population, fitness_scores)
            population = self.mutate(population)
        return best_individual

    def initialize_population(self, neural_network):
        population = []
        for _ in range(self.population_size):
            individual = NeuralNetwork(neural_network.input_size, neural_network.hidden_sizes, neural_network.output_size)
            population.append(individual)
        return population

    def calculate_fitness(self, neural_network, population, X, y):
        fitness_scores = []
        for individual in population:
            y_hat = individual.forward(X)
            fitness = np.mean(np.square(y - y_hat))
            fitness_scores.append(fitness)
        return np.array(fitness_scores)

    def crossover(self, population, fitness_scores):
        new_population = []
        fitness_probs = fitness_scores / np.sum(fitness_scores)
        for _ in range(self.population_size):
            parent1 = population[self.select_parent_index(fitness_probs)]
            parent2 = population[self.select_parent_index(fitness_probs)]
            child = self.perform_crossover(parent1, parent2)
            new_population.append(child)
        return new_population

    def select_parent_index(self, fitness_probs):
        return np.random.choice(range(self.population_size), p=fitness_probs)

    def perform_crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_sizes, parent1.output_size)
        for key in parent1.parameters.keys():
            if np.random.rand() < 0.5:
                child.parameters[key] = parent1.parameters[key]
            else:
                child.parameters[key] = parent2.parameters[key]
        return child

    def mutate(self, population):
        for individual in population:
            for key in individual.parameters.keys():
                if np.random.rand() < self.mutation_rate:
                    individual.parameters[key] += np.random.randn(*individual.parameters[key].shape) * 0.1
        return population

# Example usage
# Define your input and output data
X = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
              [2, 3, 4, 5, 6, 7, 8, 9],
              [3, 4, 5, 6, 7, 8, 9, 10],
              ...])  # Add more input samples as needed

y = np.array([[1, 0, 0, 0],  # Corresponding output for the first input sample
              [0, 1, 0, 0],  # Corresponding output for the second input sample
              [0, 0, 1, 0],  # Corresponding output for the third input sample
              ...])  # Add more output samples as needed

# Rest of the code
input_size = 8
output_size = 4
hidden_sizes = [16, 16]
population_size = 50
mutation_rate = 0.1
num_generations = 100

neural_network = NeuralNetwork(input_size, hidden_sizes, output_size)
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)
best_individual = genetic_algorithm.evolve(neural_network, X, y, num_generations)
