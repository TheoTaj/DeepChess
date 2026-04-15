import torch 
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    # by default we used the parameters from the paper. 
    def __init__(self,
                 in_channels=18,
                 conv_filters=[20, 50],
                 conv_kernels=[5, 3],
                 fc_dim=[500],
                 dropout=0.3,
                 activation_layer=nn.ELU
                ):
        """
        Instance of the ChessCNN model. Input tensor X of size C*H*W.
        Kernel w_k size C*h*w

        Args:
            in_channels (int): Number of input channels C
            conv_filters (list of int): Number of kernels for each conv layers
            conv_kernels (list of int): Size of the kernels for each conv layers (h=w)
            fc_dim (list of int): Number of neurons for each fully connected layer
            dropout (float): Dropout rate for the fully connected layers
            activation_layer (nn.Module): Activation function to use after each conv layer
        """
        super().__init__()

        self.hyperparams = {
            "conv_filters": conv_filters,
            "conv_kernels": conv_kernels,
            "fc_layers": fc_dim,
            "dropout_p": dropout,
            "activation": activation_layer.__name__ # On garde juste le nom de la fonction
        }

        # 1. Convolutioal part
        layers = []
        current_channels = in_channels
        for out_channels, kernel_size in zip(conv_filters, conv_kernels):
            layers.append(nn.Conv2d(current_channels, out_channels, 
                            kernel_size=kernel_size, padding=kernel_size//2)) # padding = same, we don't tune this "hyperparameter".
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_layer())
            layers.append(nn.Dropout2d(p=dropout)) # turns off entire channels
            current_channels = out_channels
        self.conv_block = nn.Sequential(*layers)

        # 2. MLP part

        self.flatten = nn.Flatten() # CCN outputs a tensor but MLP wants a vector
        input_size = current_channels * 8 * 8
        mlp_modules = []
        for hidden_dim in fc_dim:
            mlp_modules.append(nn.Linear(input_size, hidden_dim))
            mlp_modules.append(activation_layer())
            mlp_modules.append(nn.Dropout(p=dropout)) # turns off individual neurons
            input_size_mlp = hidden_dim

        mlp_modules.append(nn.Linear(input_size_mlp, 1))
        self.fc_block = nn.Sequential(*mlp_modules)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x

    def get_config(self):
        return self.hyperparams

if __name__ == "__main__":

    model = ChessCNN()

    dummy_input = torch.zeros((1, 18, 8, 8))
    
    try:
        output = model(dummy_input)
        print("✅ Succès ! Le flux de données traverse tout le réseau.")
        print(f"Forme de la sortie : {output.shape}") # Devrait être [1, 1]
    except Exception as e:
        print(f"❌ Erreur pendant le forward pass : {e}")
