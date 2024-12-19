import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output


def is_hermitian(matrix, tolerance=1e-10):
    """Check if a matrix is Hermitian within numerical tolerance"""
    return np.allclose(matrix, matrix.conj().T, atol=tolerance)


def is_stoquastic(matrix, tolerance=1e-10):
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag_elements = matrix[off_diag_mask]
    return np.all(np.real(off_diag_elements) <= tolerance)


def generate_valid_matrix_pair(size, max_attempts=100):
    """Generate a valid H, C matrix pair with an improved approach"""
    # Start with a stoquastic matrix H' (all off-diagonal elements <= 0)
    H_prime = np.zeros((size, size), dtype=complex)

    # Fill diagonal with random real values
    np.fill_diagonal(H_prime, np.random.rand(size))

    # Fill upper triangle with negative random values
    for i in range(size):
        for j in range(i + 1, size):
            val = -np.random.rand()
            H_prime[i, j] = val
            H_prime[j, i] = val  # Ensure Hermiticity

    # Generate a random unitary matrix C
    C, _ = np.linalg.qr(np.random.rand(size, size) + 1j * np.random.rand(size, size))

    # Transform back to get H = C† H' C
    H = C.conj().T @ H_prime @ C

    # Verify the matrices
    assert is_hermitian(H) and is_hermitian(H_prime) and is_stoquastic(H_prime)

    return H, C


def reconstruct_matrix(flat_array, size=4):
    """Reconstruct a square matrix from flattened array"""
    return flat_array.reshape(size, size)


def calculate_accuracy(model, test_H, test_H_orig, test_C_orig, train_H_mean, train_H_std, tolerance=1e-8):
    """Calculate accuracy based on stoquasticity of transformed matrices"""
    # Denormalize the test inputs
    test_H_denorm = test_H * train_H_std + train_H_mean

    # Get model predictions
    predicted_C_flat = model.forward(test_H)

    correct = 0
    total = len(test_H)

    for i in range(total):
        # Reconstruct matrices
        H = test_H_orig[i]
        C_pred = reconstruct_matrix(predicted_C_flat[i])

        # Make C_pred unitary through QR decomposition
        C_pred_unitary, _ = np.linalg.qr(C_pred + 1j * np.zeros_like(C_pred))

        # Calculate transformed H
        H_transformed = C_pred_unitary @ H @ C_pred_unitary.conj().T

        # Check if result is stoquastic and hermitian
        if is_stoquastic(H_transformed, tolerance) and is_hermitian(H_transformed, tolerance):
            correct += 1

    accuracy = correct / total
    return accuracy


def plot_accuracy(accuracies, current_epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, current_epoch + 2), accuracies, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Epoch')
    plt.grid(True)
    plt.pause(0.1)
    clear_output(wait=True)
    plt.show()


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def backward(self, X, Y, output, learning_rate):
        m = X.shape[0]
        dZ2 = output - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, Y, epochs, learning_rate, test_H, test_H_orig, test_C_orig, train_H_mean, train_H_std):
        accuracies = []
        plt.ion()  # Turn on interactive mode

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output, learning_rate)
            loss = np.mean((output - Y) ** 2)

            # Calculate and store accuracy
            accuracy = calculate_accuracy(self, test_H, test_H_orig, test_C_orig, train_H_mean, train_H_std)
            accuracies.append(accuracy * 100)

            # Plot accuracy
            plot_accuracy(accuracies, epoch)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}, Accuracy: {accuracy * 100:.2f}%")

        plt.ioff()  # Turn off interactive mode
        return accuracies


def main():
    # Parameters
    N = 10000  # Number of examples
    size = 4  # Matrix size for two qubits

    # Generate dataset with verification
    H_matrices = []
    C_matrices = []

    print("Generating and verifying matrices...")
    for i in range(N):
        try:
            H, C = generate_valid_matrix_pair(size)
            H_matrices.append(H)
            C_matrices.append(C)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{N} valid matrix pairs")
        except ValueError as e:
            print(f"Error generating matrix pair {i}: {e}")
            continue

    if len(H_matrices) < N:
        print(f"Warning: Only generated {len(H_matrices)} valid matrix pairs out of {N} requested")

    dataset = list(zip(H_matrices, C_matrices))

    # Split dataset into train and test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Extract features and labels for training and testing
    train_H = np.array([np.real(h.flatten()) for h, _ in train_data])
    train_C = np.array([np.real(c.flatten()) for _, c in train_data])
    test_H = np.array([np.real(h.flatten()) for h, _ in test_data])
    test_C = np.array([np.real(c.flatten()) for _, c in test_data])

    # Normalize the training data
    train_H_mean = train_H.mean(axis=0)
    train_H_std = train_H.std(axis=0)
    train_H = (train_H - train_H_mean) / train_H_std
    test_H = (test_H - train_H_mean) / train_H_std

    # Model parameters
    input_size = train_H.shape[1]
    hidden_size = 64
    output_size = train_C.shape[1]

    # Initialize and train the model
    print("\nTraining model...")
    model = SimpleMLP(input_size, hidden_size, output_size)

    # Get original test matrices for accuracy calculation
    test_H_orig = [h for h, _ in test_data]
    test_C_orig = [c for _, c in test_data]

    # Train with accuracy monitoring
    accuracies = model.train(train_H, train_C, epochs=200, learning_rate=0.001,
                             test_H=test_H, test_H_orig=test_H_orig,
                             test_C_orig=test_C_orig,
                             train_H_mean=train_H_mean, train_H_std=train_H_std)

    # Final evaluation
    predictions = model.forward(test_H)
    test_loss = np.mean((predictions - test_C) ** 2)
    final_accuracy = calculate_accuracy(model, test_H, test_H_orig, test_C_orig, train_H_mean, train_H_std)

    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss}")
    print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
