import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

def download_dataset(df_name="mateuszgrzyb/lichess-stockfish-normalized", output_dir="data", filename= "dataset", nrows: int = 5_000_000):
    """
    Creates a parquet file containing the dataset with the specified number of rows. Uses a fixed random seed for reproducibility. 
    The dataset is shuffled in order to have a representative dataset.
    Args:
        df_name (str): The name of the dataset to load from Hugging Face.
        output_dir (str): The directory where the parquet file will be saved.
        filename (str): The name of the parquet file (without extension).
        nrows (int): The number of rows to include in the parquet file.
    """

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{filename}.parquet")

    dataset = load_dataset(df_name, split="train", streaming=True, token=hf_token)

    dataset = dataset.shuffle(seed=42, buffer_size=100000) # changing the buffer size will change the result !

    print(f"Loading dataset '{df_name}' for {nrows} rows...'")
    subset = dataset.take(nrows)
    print("Dataset loaded, converting to DataFrame...")
    data_list = list(subset)
    df = pd.DataFrame(data_list)

    if 'depth' in df.columns:
        df = df.drop(columns=['depth'])

    df = df.dropna(subset=['cp', 'mate'], how='all')

    print(f"Saving {len(df)} rows in {full_path}...")
    df.to_parquet(full_path, compression='snappy', index=False)

if __name__ == "__main__":
    nrows = 5_000_000 # then change it for 5_000_000
    filename = f"dataset_{nrows}"
    download_dataset(filename=filename,nrows=nrows)