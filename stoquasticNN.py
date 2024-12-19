import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch


# Device configuration
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


def is_hermitian(matrix, tolerance=1e-10):
    """Check if a matrix is Hermitian within numerical tolerance"""
    return np.allclose(matrix, matrix.conj().T, atol=tolerance)


def is_stoquastic(matrix, tolerance=1e-10):
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag_elements = matrix[off_diag_mask]
    return np.all(np.real(off_diag_elements) <= tolerance)


def generate_valid_matrix_pair(size):
    """Generate a valid H, C matrix pair"""
    H_prime = np.zeros((size, size), dtype=complex)
    np.fill_diagonal(H_prime, np.random.rand(size))

    for i in range(size):
        for j in range(i + 1, size):
            val = -np.random.rand()
            H_prime[i, j] = val
            H_prime[j, i] = val

    C, _ = np.linalg.qr(np.random.rand(size, size) + 1j * np.random.rand(size, size))
    H = C.conj().T @ H_prime @ C

    assert is_hermitian(H) and is_hermitian(H_prime) and is_stoquastic(H_prime)
    return H, C


class ComplexMLP:
    def __init__(self, input_size, output_size):
        self.layers = []
        layer_sizes = [input_size, 256, 512, 512, 256, output_size]

        for i in range(len(layer_sizes) - 1):
            layer = {
                'W': torch.randn(layer_sizes[i], layer_sizes[i + 1], device=device) / np.sqrt(layer_sizes[i]),
                'b': torch.zeros(layer_sizes[i + 1], device=device),
                'gamma': torch.ones(layer_sizes[i + 1], device=device),
                'beta': torch.zeros(layer_sizes[i + 1], device=device),
                'running_mean': torch.zeros(layer_sizes[i + 1], device=device),
                'running_var': torch.ones(layer_sizes[i + 1], device=device)
            }
            self.layers.append(layer)

    def batch_normalize(self, x, layer, training=True):
        if training:
            mean = torch.mean(x, dim=0)
            var = torch.var(x, dim=0, unbiased=False) + 1e-5

            layer['running_mean'] = 0.9 * layer['running_mean'] + 0.1 * mean
            layer['running_var'] = 0.9 * layer['running_var'] + 0.1 * var
        else:
            mean = layer['running_mean']
            var = layer['running_var']

        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        out = layer['gamma'] * x_norm + layer['beta']

        if training:
            layer['cache'] = (x_norm, var)
        return out

    def forward(self, x, training=True):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        self.activations = [x]

        for i, layer in enumerate(self.layers[:-1]):
            z = torch.mm(self.activations[-1], layer['W']) + layer['b']
            h = self.batch_normalize(z, layer, training)
            h = torch.tanh(h)

            if i > 0 and h.shape == self.activations[-2].shape:
                h += self.activations[-2]

            self.activations.append(h)

        z = torch.mm(self.activations[-1], self.layers[-1]['W']) + self.layers[-1]['b']
        self.activations.append(z)

        return self.activations[-1].cpu().numpy() if not training else self.activations[-1]

    def backward(self, x, y, learning_rate=0.001):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        m = x.shape[0]  # Batch size
        grad = self.activations[-1] - y  # Gradient of the loss w.r.t. output

        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                # For the last layer
                dW = torch.mm(self.activations[i].t(), grad)
                db = torch.sum(grad, dim=0)
            else:
                # For all other layers
                grad = torch.mm(grad, self.layers[i + 1]['W'].t()) * (1 - torch.tanh(self.activations[i + 1]) ** 2)

                # Skip connection logic: Only add if shapes match
                if i > 0 and grad.shape == self.activations[i - 1].shape:
                    grad += self.activations[i - 1]

                dW = torch.mm(self.activations[i].t(), grad)
                db = torch.sum(grad, dim=0)

            # Momentum updates
            if not hasattr(self.layers[i], 'mW'):
                self.layers[i]['mW'] = torch.zeros_like(self.layers[i]['W'])
                self.layers[i]['mb'] = torch.zeros_like(self.layers[i]['b'])

            self.layers[i]['mW'] = 0.9 * self.layers[i]['mW'] - learning_rate * dW / m
            self.layers[i]['mb'] = 0.9 * self.layers[i]['mb'] - learning_rate * db / m

            # Apply updates
            self.layers[i]['W'] += self.layers[i]['mW']
            self.layers[i]['b'] += self.layers[i]['mb']


def plot_accuracy(accuracies, current_epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, current_epoch + 2), accuracies, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Accuracy vs Epoch (Current: {accuracies[-1]:.2f}%)')
    plt.grid(True)
    plt.pause(0.1)
    clear_output(wait=True)
    plt.show()


def main():
    # Parameters
    N = 10000
    size = 4

    print("Generating and verifying matrices...")
    H_matrices = []
    C_matrices = []

    for i in range(N):
        try:
            H, C = generate_valid_matrix_pair(size)
            H_matrices.append(H)
            C_matrices.append(C)
            # if (i + 1) % 10 == 0:
            #     print(f"Generated {i + 1}/{N} valid matrix pairs")
        except ValueError as e:
            print(f"Error generating matrix pair {i}: {e}")
            continue

    dataset = list(zip(H_matrices, C_matrices))

    # Split and prepare data
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Extract features and labels
    train_H = np.array([np.concatenate([h.real.flatten(), h.imag.flatten()]) for h, _ in train_data])
    train_C = np.array([np.concatenate([c.real.flatten(), c.imag.flatten()]) for _, c in train_data])
    test_H = np.array([np.concatenate([h.real.flatten(), h.imag.flatten()]) for h, _ in test_data])
    test_C = np.array([np.concatenate([c.real.flatten(), c.imag.flatten()]) for _, c in test_data])

    # Normalize data
    train_H_mean = train_H.mean(axis=0)
    train_H_std = train_H.std(axis=0) + 1e-8
    train_H = (train_H - train_H_mean) / train_H_std
    test_H = (test_H - train_H_mean) / train_H_std

    # Initialize model
    input_size = train_H.shape[1]
    output_size = train_C.shape[1]
    model = ComplexMLP(input_size, output_size)

    # Training loop
    epochs = 1000
    accuracies = []
    plt.ion()

    test_H_orig = [h for h, _ in test_data]
    test_C_orig = [c for _, c in test_data]

    print("\nTraining model...")
    batch_size = 32
    n_batches = len(train_H) // batch_size

    for epoch in range(epochs):
        # Mini-batch training
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_H = train_H[start_idx:end_idx]
            batch_C = train_C[start_idx:end_idx]

            output = model.forward(batch_H)
            model.backward(batch_H, batch_C, learning_rate=0.0001)

        # Calculate accuracy
        predictions = model.forward(test_H, training=False)
        correct = 0
        total = len(test_H)

        for i in range(total):
            H = test_H_orig[i]
            C_pred = predictions[i][:size * size].reshape(size, size) + 1j * predictions[i][size * size:].reshape(size,
                                                                                                                  size)
            C_pred_unitary, _ = np.linalg.qr(C_pred)
            H_transformed = C_pred_unitary @ H @ C_pred_unitary.conj().T

            if is_stoquastic(H_transformed) and is_hermitian(H_transformed):
                correct += 1

        accuracy = (correct / total) * 100
        accuracies.append(accuracy)

        plot_accuracy(accuracies, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%")

    plt.ioff()
    print(f"\nFinal Test Accuracy: {accuracies[-1]:.2f}%")


if __name__ == "__main__":
    main()
