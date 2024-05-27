import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

    

    
def combine_one_hot_pairs(tensor):
    # Transpose the tensor
    tensor = tensor.transpose(1, 2)  # (batch_size, 8, 10) -> (batch_size, 10, 8)

    batch_size, num_classes, num_rows = tensor.shape
    assert num_rows % 2 == 0, "Number of rows must be even"

    num_pairs = num_rows // 2
    combined_dim = num_classes ** 2

    # Initialize the combined tensor
    combined_tensor = torch.zeros(batch_size, num_pairs, combined_dim)

    for batch_idx in range(batch_size):
        for pair_idx in range(num_pairs):
            # Extract pairs of rows
            row1 = tensor[batch_idx, :, 2 * pair_idx]
            row2 = tensor[batch_idx, :, 2 * pair_idx + 1]

            # Find the indices of the one-hot encodings
            index1 = row1.argmax().item()
            index2 = row2.argmax().item()
            # print(f"index1={index1} ")
            # Calculate the combined index
            combined_index = num_classes * index1 + index2

            # Set the corresponding position in the combined one-hot encoding
            combined_tensor[batch_idx, pair_idx, combined_index] = 1
    print(f"combined_tensor.shape={combined_tensor.shape}")
    return combined_tensor

def combine_one_hot_pairs(tensor):
    # print(f"original shape={tensor.shape}")
    # Transpose the tensor
    tensor = tensor.transpose(1, 2)  # (batch_size, 8, 10) -> (batch_size, 10, 8)
    
    batch_size, num_classes, num_rows = tensor.shape
    assert num_rows % 2 == 0, "Number of rows must be even"

    num_pairs = num_rows // 2
    combined_dim = num_classes ** 2

    # Initialize the combined tensor
    combined_tensor = torch.zeros(batch_size, num_pairs, combined_dim,device=tensor.device)


    for pair_idx in range(num_pairs):
        # Extract pairs of rows
        row1 = tensor[:, :, 2 * pair_idx]
        row2 = tensor[:, :, 2 * pair_idx + 1]

        # Find the indices of the one-hot encodings
        index1 = row1.argmax(dim=1)
        index2 = row2.argmax(dim=1)
        # print(f"index.shape={index1}")
        # Calculate the combined index
        combined_index = num_classes * index1 + index2

        # Set the corresponding position in the combined one-hot encoding
        combined_tensor[torch.arange(batch_size), pair_idx, combined_index] = 1
    # print(f"combined_tensor.shape={combined_tensor[0]}")
    return combined_tensor
    
    
    
     
class NonOverlappingLocallyConnected1d_new(nn.Module):
    def __init__(self, input_channels, out_channels, out_dim, bias=False, s=2,num_layers=1):
        super(NonOverlappingLocallyConnected1d_new, self).__init__()
        self.s = s
        self.weight = nn.Parameter(torch.randn(1 * s**(num_layers-1),input_channels, out_channels ))
        # print(f"self.weight.shape={self.weight.shape}")


        self.input_channels = input_channels

    def forward(self, x):
        batch_size = x.size(0)
        result = x[:, :, :, None] * self.weight[None, :, :, :]  # Shape: (256, 2, 100, 10)

        # Sum along the third dimension to get the final result
        result = result.sum(dim=2)  
        result /= self.input_channels ** .5
        x=result
        x=x.sum(dim=[1])
        return x

class patchwise_perceptron(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, bias=False, s=2):
        super(patchwise_perceptron, self).__init__()
        d = s ** num_layers
        self.s = s
        self.num_layers=num_layers
        self.hier = nn.Sequential(
            NonOverlappingLocallyConnected1d_new(input_channels, h, d // s, bias, s,num_layers),
            # nn.ReLU(),
        )

    def forward(self, x):
        x=x.transpose(2,1)
        x=combine_one_hot_pairs(x)
        y = self.hier(x)
        y = y.float()
        # y = y @ self.beta / self.beta.size(0)
        return y

    def visualize_weights(self, filename='weights.png'):
        combined_weight = None
        for layer in self.hier:
            if isinstance(layer, NonOverlappingLocallyConnected1d_new):
                weight = layer.weight.data.cpu().numpy()
                weight = weight.reshape(weight.shape[0], -1)  # Flatten the locally connected dimensions
                if combined_weight is None:
                    combined_weight = weight
                else:
                    combined_weight = np.dot(combined_weight, weight.T)

        if combined_weight is not None:
            combined_weight=combined_weight.reshape(self.s**(self.num_layers-1),100,10)
            combined_weight_0 = combined_weight[0, :, :]
            # print each sum of columns where element bigger than 3 that are bigger than 10
            for row in combined_weight_0.T:
            # Filter elements in the row that are greater than 3
                filtered_elements = row[row > 3]
                row_sum = np.sum(filtered_elements)
                if row_sum > 10:
                    print(row_sum)
            combined_weight_1 = combined_weight[1, :, :]
            
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            
            im0 = axs[0].imshow(combined_weight_0, cmap='viridis', aspect='auto')
            axs[0].set_title('Weight of patch position 1')
            axs[0].set_xlabel('Class label')
            axs[0].set_ylabel('One hot encoded s-tuple')
            
            im1 = axs[1].imshow(combined_weight_1, cmap='viridis', aspect='auto')
            axs[1].set_title('Weight of patch position 2')
            axs[1].set_xlabel('Class label')
            axs[1].set_ylabel('One hot encoded s-tuple')
            
            # Add colorbars
            cbar0 = fig.colorbar(im0, ax=axs[0])
            cbar0.set_label('Weight Value')
            cbar1 = fig.colorbar(im1, ax=axs[1])
            cbar1.set_label('Weight Value')
            
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

