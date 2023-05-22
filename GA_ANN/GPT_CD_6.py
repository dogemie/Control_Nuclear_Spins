import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        self.parameters = {}
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            layer_in = layer_sizes[i-1]
            layer_out = layer_sizes[i]
            self.parameters[f"W{i}"] = np.random.randn(layer_in, layer_out)
            self.parameters[f"b{i}"] = np.random.randn(layer_out)

    def forward(self, X):
        self.activations = {"A0": X}

        for i in range(1, self.num_layers + 1):
            W = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]
            Z = np.dot(self.activations[f"A{i-1}"], W) + b
            A = self.activation_function(Z)
            self.activations[f"A{i}"] = A

        return self.activations[f"A{self.num_layers}"]

    def activation_function(self, Z):
        # Define your activation function, e.g., sigmoid, relu, etc.
        return 1 / (1 + np.exp(-Z))

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def evolve(self, neural_network, X, y, num_generations):
        population = self.initialize_population(neural_network)
        best_individual = None

        for generation in range(num_generations):
            fitness_scores = self.calculate_fitness(neural_network, population, X, y)
            best_individual = population[np.argmax(fitness_scores)]
            print("Generation:", generation+1, "Best Fitness:", np.max(fitness_scores))

            if np.max(fitness_scores) == 1.0:
                break

            population = self.create_next_generation(population, fitness_scores)

        return best_individual

    def initialize_population(self, neural_network):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for key, value in neural_network.parameters.items():
                individual[key] = np.random.randn(*value.shape)
            population.append(individual)
        return population

    def calculate_fitness(self, neural_network, population, X, y):
        fitness_scores = []
        for individual in population:
            neural_network.parameters = individual
            y_hat = neural_network.forward(X)
            fitness = np.mean(np.square(y - y_hat))
            fitness_scores.append(1.0 - fitness)
            #여기서 fitness를 수정해야 할 듯
        return np.array(fitness_scores)

    def create_next_generation(self, population, fitness_scores):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents(population, fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def select_parents(self, population, fitness_scores):
        # Roulette wheel selection
        probabilities = fitness_scores / np.sum(fitness_scores)
        indices = np.random.choice(len(population), size=2, replace=True, p=probabilities)
        parent1 = population[indices[0]]
        parent2 = population[indices[1]]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            mask = np.random.randint(low=0, high=2, size=parent1[key].shape).astype(bool)
            child[key] = np.where(mask, parent1[key], parent2[key])
        return child

    def mutate(self, individual):
        for key in individual.keys():
            mask = np.random.rand(*individual[key].shape) < self.mutation_rate
            mutation = np.random.randn(*individual[key].shape)
            individual[key] = np.where(mask, individual[key] + mutation, individual[key])
        return individual

# Example usage
input_size = 8
hidden_sizes = [16, 8]
output_size = 4

neural_network = NeuralNetwork(input_size, hidden_sizes, output_size)
genetic_algorithm = GeneticAlgorithm(population_size=100, mutation_rate=0.01)

# Generate training data
X = np.random.rand(300, input_size)
y = np.random.rand(300, output_size)

# Evolve the neural network
best_individual = genetic_algorithm.evolve(neural_network, X, y, num_generations=5000)
