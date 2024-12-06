import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.animation import FFMpegWriter


class SymmetricEigenvalueNet(nn.Module):
    def __init__(self, matrix_size, hidden_layers=[128, 256, 128]):
        super(SymmetricEigenvalueNet, self).__init__()

        # Input layer expects flattened symmetric matrix
        input_size = (matrix_size * (matrix_size + 1)) // 2

        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        self.input_norm = nn.LayerNorm(hidden_layers[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.hidden_layers.append(nn.LayerNorm(hidden_layers[i + 1]))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(hidden_layers[-1], matrix_size)

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Preprocess symmetric matrix to flattened lower triangular form
        x_flat = self.extract_lower_triangular(x)

        # Initial layers
        x = self.input_layer(x_flat)
        x = self.input_norm(x)
        x = F.relu(x)

        # Hidden layers with residual connections
        for i in range(0, len(self.hidden_layers), 3):
            # identity = x
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i + 1](x)
            x = self.hidden_layers[i + 2](x)
            # if x.shape == identity.shape:  # Add residual connection if shapes match
            #     print('hi')
            #     x = x + identity

        # Output layer without activation to allow negative values
        x = self.output_layer(x)
        return x

    @staticmethod
    def extract_lower_triangular(matrix):
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)
        batch_size, n, _ = matrix.shape
        mask = torch.tril(torch.ones(n, n, device=matrix.device, dtype=torch.bool))
        lower_tri = matrix[mask.repeat(batch_size, 1, 1)]
        return lower_tri.view(batch_size, -1)


def custom_eigenvalue_loss(predicted, target):
    # MSE loss
    mse_loss = F.mse_loss(predicted, target)

    # Ordering loss to maintain eigenvalue ordering
    # pred_sorted = torch.sort(predicted, dim=1)[0]
    # target_sorted = torch.sort(target, dim=1)[0]
    # ordering_loss = F.mse_loss(pred_sorted, target_sorted)
    #
    # # Combine losses
    # total_loss = mse_loss + 0.5 * ordering_loss
    return mse_loss


def train_eigenvalue_network(symmetric_matrices, true_eigenvalues, test_matrix, epochs=1000, batch_size=32):
    dataset = torch.utils.data.TensorDataset(symmetric_matrices, true_eigenvalues)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SymmetricEigenvalueNet(matrix_size=symmetric_matrices.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # Get true eigenvalues for test matrix
    test_true_eigenvalues = generate_true_eigenvalues(test_matrix)

    # Initialize plot
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(right=0.85)
    ax = fig.add_subplot(111)
    plt.ion()

    # Set up the movie writer with lower fps for slower animation
    writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    # Create prediction lines and true value lines with matching colors
    colors = plt.cm.rainbow(np.linspace(0, 1, symmetric_matrices.shape[1]))
    lines = []
    true_lines = []

    for i, color in enumerate(colors):
        line, = ax.plot([], [], label=f'Predicted λ{i + 1}', color=color)
        lines.append(line)
        true_line, = ax.plot([], [], '--', label=f'True λ{i + 1}', color=color, alpha=0.5)
        true_lines.append(true_line)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalues')
    ax.set_title('Eigenvalue Prediction Evolution During Training')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

    # Lists to store history
    epochs_list = []
    eigenvalues_history = []

    # Start recording the movie
    with writer.saving(fig, "eigenvalues_training.mp4", dpi=100):
        for epoch in range(epochs):
            epoch_loss = 0.0
            model.train()

            for batch_matrices, batch_eigenvalues in dataloader:
                optimizer.zero_grad()
                predicted_eigenvalues = model(batch_matrices)
                loss = custom_eigenvalue_loss(predicted_eigenvalues, batch_eigenvalues)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_epoch_loss)

            # Evaluate on test matrix and update plot
            model.eval()
            with torch.no_grad():
                test_pred = model(test_matrix.unsqueeze(0))[0]
                epochs_list.append(epoch)
                eigenvalues_history.append(test_pred.numpy())

            if epoch % 20 == 0:  # Update plot less frequently (every 20 epochs instead of 10)
                # Update prediction lines
                for i, line in enumerate(lines):
                    line.set_xdata(epochs_list)
                    line.set_ydata([ev[i] for ev in eigenvalues_history])
                    true_lines[i].set_xdata([0, epoch])
                    true_lines[i].set_ydata([test_true_eigenvalues[i], test_true_eigenvalues[i]])

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)  # Add small delay for smoother animation

                # Save frame to video
                writer.grab_frame()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {avg_epoch_loss}')

    plt.ioff()
    return model


def generate_symmetric_matrix(size=5, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    matrix = np.random.rand(size, size)
    symmetric_matrix = (matrix + matrix.T) / 2
    return torch.tensor(symmetric_matrix, dtype=torch.float32)


def generate_true_eigenvalues(symmetric_matrix):
    return torch.tensor(np.linalg.eigvals(symmetric_matrix.numpy()), dtype=torch.float32)


def main():
    matrix_size = 30
    num_matrices = matrix_size * 10

    # Generate training data
    symmetric_matrices = torch.stack([generate_symmetric_matrix(size=matrix_size, seed=i)
                                      for i in range(num_matrices)])
    true_eigenvalues = torch.stack([generate_true_eigenvalues(matrix)
                                    for matrix in symmetric_matrices])

    # Generate test matrix
    test_matrix = generate_symmetric_matrix(size=matrix_size, seed=42)
    test_eigenvalues = generate_true_eigenvalues(test_matrix)

    print("Test Symmetric Matrix:")
    print(test_matrix)
    print("\nTrue Eigenvalues:")
    print(test_eigenvalues)

    # Train the network with visualization
    model = train_eigenvalue_network(
        symmetric_matrices,
        true_eigenvalues,
        test_matrix,
        epochs=600,
        batch_size=32
    )

    # Final prediction
    model.eval()
    with torch.no_grad():
        predicted_eigenvalues = model(test_matrix.unsqueeze(0))
        print("\nFinal Predicted Eigenvalues:")
        print(predicted_eigenvalues[0])
        mse = F.mse_loss(predicted_eigenvalues[0], test_eigenvalues)
        print(f"\nFinal Mean Squared Error: {mse.item()}")


if __name__ == "__main__":
    main()
