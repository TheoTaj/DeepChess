import torch
import pandas as pd
import numpy as np
import chess
import sys
import os
import time

def run_tests():
    print("="*50)
    print("🚀 DÉBUT DU TEST DE L'ENVIRONNEMENT SUR ALAN")
    print("="*50)

    # 1. Tests Système
    print(f"\n[1] SYSTÈME")
    print(f"    - Python version: {sys.version.split()[0]}")
    print(f"    - PyTorch version: {torch.__version__}")
    print(f"    - Pandas version: {pd.__version__}")

    # 2. Tests GPU (Le plus important)
    print(f"\n[2] GPU / CUDA")
    cuda_available = torch.cuda.is_available()
    print(f"    - CUDA disponible: {cuda_available}")
    
    if cuda_available:
        print(f"    - Nombre de GPUs: {torch.cuda.device_count()}")
        print(f"    - Nom du GPU: {torch.cuda.get_device_name(0)}")
        
        # Test de calcul réel sur GPU
        try:
            start_time = time.time()
            x = torch.randn(2000, 2000).cuda()
            y = torch.randn(2000, 2000).cuda()
            z = torch.matmul(x, y)
            torch.cuda.synchronize() # Attendre la fin du calcul
            duration = time.time() - start_time
            print(f"    - Test de calcul (matrice 2000x2000): RÉUSSI ({duration:.4f}s)")
        except Exception as e:
            print(f"    - ❌ ERREUR lors du calcul GPU: {e}")
    else:
        print("    - ⚠️ ALERTE: Torch ne voit pas le GPU. Vérifiez les drivers NVIDIA ou la version de Torch.")

    # 3. Tests Data
    print(f"\n[3] DATASET")
    path = "data/dataset_1000.parquet"
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            print(f"    - Fichier '{path}' chargé: RÉUSSI")
            print(f"    - Nombre de lignes: {len(df)}")
            print(f"    - Colonnes présentes: {list(df.columns)}")
        except Exception as e:
            print(f"    - ❌ ERREUR lors de la lecture du Parquet: {e}")
    else:
        print(f"    - ❌ ERREUR: Le fichier '{path}' est introuvable.")

    # 4. Tests Logique Échecs
    print(f"\n[4] LOGIQUE ÉCHECS (python-chess)")
    try:
        board = chess.Board()
        print(f"    - Création d'un plateau: RÉUSSI")
        print(f"    - FEN initial: {board.fen()}")
        # Test d'une règle spécifique (Roque possible ?)
        print(f"    - Droit au roque blanc (kingside): {board.has_kingside_castling_rights(chess.WHITE)}")
    except Exception as e:
        print(f"    - ❌ ERREUR avec python-chess: {e}")

    print("\n" + "="*50)
    print("✅ FIN DES TESTS")
    print("="*50)

if __name__ == "__main__":
    run_tests()