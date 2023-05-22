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
            a = self.softmax(z)
        return a

    def softmax(self, z):
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

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
            self.mutate(offspring)
            population.append(offspring)

        for generation in range(self.population_size):
            fitness_scores = []
            for individual in population:
                y_hat = individual.forward(X)
                fitness = self.calculate_fitness(y_hat, y, neural_network.output_size)
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
        for weight_matrix in individual.weights:
            for weight in weight_matrix.flat:
                if np.random.random() < self.mutation_rate:
                    weight += np.random.randn()
        for bias_vector in individual.biases:
            for bias in bias_vector.flat:
                if np.random.random() < self.mutation_rate:
                    bias += np.random.randn()

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
                                      parent1.hidden_sizes,
                                      parent1.output_size)
            offspring.weights = [np.where(np.random.random(weight_matrix.shape) < 0.5,
                                           parent1.weights[i],
                                           parent2.weights[i])
                                 for i, weight_matrix in enumerate(offspring.weights)]
            offspring.biases = [np.where(np.random.random(bias_vector.shape) < 0.5,
                                          parent1.biases[i],
                                          parent2.biases[i])
                                for i, bias_vector in enumerate(offspring.biases)]
            offspring_population.append(offspring)

        return offspring_population

    def calculate_fitness(self, y_hat, y, output_size):
        return np.mean(np.square(y - y_hat[:, :output_size]))

# Pathfinding problem example
class PathfindingProblem:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.start_position = (0, 0)
        self.goal_position = (grid_size - 1, grid_size - 1)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.grid[self.start_position] = 1
        self.grid[self.goal_position] = 2

    def generate_training_data(self, num_samples):
        X = []
        y = []
        for _ in range(num_samples):
            path = self.find_path()
            X.append(self.grid.flatten())
            y.append(self.encode_path_directions(path))
        X = np.array(X)
        y = np.array(y)
        return X, y

    def find_path(self):
        current_position = self.start_position
        path = []
        while current_position != self.goal_position:
            next_position = self.choose_next_position(current_position)
            path.append(next_position)
            current_position = next_position
        return path

    def choose_next_position(self, current_position):
        x, y = current_position
        possible_moves = []
        if x < self.grid_size - 1:
            possible_moves.append((x + 1, y))  # Move east
        if x > 0:
            possible_moves.append((x - 1, y))  # Move west
        if y < self.grid_size - 1:
            possible_moves.append((x, y + 1))  # Move south
        if y > 0:
            possible_moves.append((x, y - 1))  # Move north
        return possible_moves[np.random.randint(len(possible_moves))]

    def encode_path_directions(self, path):
        encoded_path = np.zeros((len(path), 4))
        for i, position in enumerate(path):
            x, y = position
            if x > 0 and (x - 1, y) == path[i - 1]:
                encoded_path[i][0] = 1  # Move west
            elif x < self.grid_size - 1 and (x + 1, y) == path[i - 1]:
                encoded_path[i][1] = 1  # Move east
            elif y > 0 and (x, y - 1) == path[i - 1]:
                encoded_path[i][2] = 1  # Move north
            elif y < self.grid_size - 1 and (x, y + 1) == path[i - 1]:
                encoded_path[i][3] = 1  # Move south
        return encoded_path

# Example usage
grid_size = 5
population_size = 100
mutation_rate = 0.01
num_generations = 100

pathfinding_problem = PathfindingProblem(grid_size)
num_samples = 100  # Number of training samples
X, y = pathfinding_problem.generate_training_data(num_samples)

neural_network = NeuralNetwork(grid_size * grid_size, [16], 4)
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)

for generation in range(num_generations):
    best_individual = genetic_algorithm.evolve(neural_network, X, y)
    predicted_path = pathfinding_problem.encode_path_directions(pathfinding_problem.find_path())
    predicted_directions = best_individual.forward(pathfinding_problem.grid.flatten())
    print("Generation:", generation)
    print("Predicted Directions:", predicted_directions)
    print("Actual Directions:", predicted_path)
    print("")
