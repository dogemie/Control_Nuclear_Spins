import numpy as np

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        # Initialize weights and biases for each layer
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        num_layers = len(layer_sizes)
        for i in range(1, num_layers):
            # Randomly initialize weights and biases for each layer
            weight_matrix = np.random.randn(layer_sizes[i - 1], layer_sizes[i])
            bias_vector = np.random.randn(layer_sizes[i])
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, X):
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
        return a

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# Define the genetic algorithm class
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def evolve(self, neural_network, X, y):
        population = []
        for _ in range(self.population_size):
            offspring = NeuralNetwork(neural_network.input_size,
                                      neural_network.hidden_sizes,
                                      neural_network.output_size)
            offspring.weights1 = neural_network.weights1 + np.random.randn(*neural_network.weights1.shape)
            offspring.weights2 = neural_network.weights2 + np.random.randn(*neural_network.weights2.shape)
            self.mutate(offspring)
            population.append(offspring)

        for generation in range(self.population_size):
            fitness_scores = []
            for individual in population:
                y_hat = individual.forward(X)
                fitness = self.calculate_fitness(y_hat, y)
                fitness_scores.append(fitness)

            # Select parents based on fitness scores
            parents = self.selection(population, fitness_scores)

            # Generate offspring through crossover
            offspring_population = self.crossover(parents)

            # Mutate offspring population
            for individual in offspring_population:
                self.mutate(individual)

            # Replace the previous generation with offspring population
            population = offspring_population

        # Return the best-performing individual
        best_individual = max(population, key=lambda individual: self.calculate_fitness(individual.forward(X), y))
        return best_individual

    def mutate(self, individual):
        for weight in individual.weights1.flat:
            if np.random.random() < self.mutation_rate:
                weight += np.random.randn()
        for weight in individual.weights2.flat:
            if np.random.random() < self.mutation_rate:
                weight += np.random.randn()

    def selection(self, population, fitness_scores):
        fitness_sum = np.sum(fitness_scores)
        probabilities = fitness_scores / fitness_sum
        parents = np.random.choice(population, size=self.population_size, replace=True, p=probabilities)
        return parents

    def crossover(self, parents):
        offspring_population = []
        for i in range(self.population_size):
            parent1 = parents[i]
            parent2 = parents[np.random.randint(self.population_size)]
            offspring = NeuralNetwork(parent1.input_size,
                                      parent1.hidden_size,
                                      parent1.output_size)
            offspring.weights1 = np.where(np.random.random(offspring.weights1.shape) < 0.5,
                                          parent1.weights1,
                                          parent2.weights1)
            offspring.weights2 = np.where(np.random.random(offspring.weights2.shape) < 0.5,
                                          parent1.weights2,
                                          parent2.weights2)
            offspring_population.append(offspring)
        return offspring_population

    def calculate_fitness(self, y_hat, y):
        # Replace with an appropriate fitness function for your problem
        return np.mean(np.square(y - y_hat))

# Example usage
X = np.random.randn(1, 8)  # Example input with shape (1, 8)
hidden_sizes = [4, 4]  # Specify the sizes of hidden layers as a list
neural_network = NeuralNetwork(8, hidden_sizes, 4)  # Create the neural network with input size 8, 2 hidden layers of size 4 each, and output size 4
output = neural_network.forward(X)
y = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

population_size = 200
mutation_rate = 0.01

genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)

best_individual = genetic_algorithm.evolve(neural_network, X, y)
print("Best individual's weights:")
print("Layer 1:")
print(best_individual.weights1)
print("Layer 2:")
print(best_individual.weights2)
print("X:")
print(best_individual.forward(X))
print("y:")
print(y)
print("Output:")
print(output)