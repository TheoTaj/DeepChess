import torch
import torch.onnx
from CNN import ChessCNN
from ViT import ChessViT

def convert_pth_to_onnx(pth_path, onnx_path, conv_filters, fc_layers):
    device = torch.device("cpu")
    model = ChessCNN(conv_filters=conv_filters, fc_dim=fc_layers)
    
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    dummy_input = torch.randn(1, 18, 8, 8)
    
    print(f"Conversion de {pth_path} vers {onnx_path}...")
    torch.onnx.export(
        model,                  # Le modèle à convertir
        dummy_input,            # Un exemple d'input pour "tracer" le graphe
        onnx_path,              # Nom du fichier de sortie
        export_params=True,     # Exporte les poids entraînés
        opset_version=17,       # Version stable et moderne d'ONNX
        do_constant_folding=True, # Optimisation : fusionne les constantes
        input_names=['input'],   # Nom de l'entrée pour onnxruntime
        output_names=['output'], # Nom de la sortie
        dynamic_axes={'input' : {0 : 'batch_size'}, # Permet de changer la taille du batch plus tard
                      'output' : {0 : 'batch_size'}}
    )
    print("Exportation terminée avec succès !")

def convert_vit_pth_to_onnx(pth_path, onnx_path, embed_dim=128, n_blocks=4, n_heads=4, mlp_dim=256):
    device = torch.device("cpu")
    model = ChessViT(embed_dim=embed_dim, n_blocks=n_blocks, n_heads=n_heads, mlp_dim=mlp_dim)

    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    dummy_input = torch.randn(1, 18, 8, 8)
    print(f"Conversion de {pth_path} vers {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )
    print("Exportation terminée avec succès !")


if __name__ == "__main__":
    # Paramètres à adapter selon tes modèles
    CONV_FILTERS = [64, 128, 256]
    FC_LAYERS = [512, 256, 128]
    
    # convert_pth_to_onnx(
    #     pth_path="models/CNN_ASYM_3.pth", 
    #     onnx_path="models/CNN_ASYM.onnx",
    #     conv_filters=CONV_FILTERS,
    #     fc_layers=FC_LAYERS
    # )
    convert_vit_pth_to_onnx(
        pth_path="models/ViT3.pth", 
        onnx_path="models/ViT_SYM.onnx",
        embed_dim=384,
        n_blocks=8,
        n_heads=8,
        mlp_dim=768
    )