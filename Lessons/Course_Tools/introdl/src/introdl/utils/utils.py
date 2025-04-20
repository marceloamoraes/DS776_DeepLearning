import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms.v2 as transforms
import sys
import os
import random
import numpy as np
import pandas as pd
import inspect
from torchinfo import summary
import traceback
from textwrap import TextWrapper
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
import warnings
import shutil
from IPython.display import display, IFrame
from IPython.core.display import HTML
import gc
import nbformat
import nbformat
from nbformat import validate
from nbformat.validator import NotebookValidationError
import subprocess
import tempfile
import shutil
import inspect

# Fallback-safe normalize import
try:
    from nbformat.normalized import normalize
except ImportError:
    def normalize(nb): return nb  # no-op if normalize not available


try:
    import dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
finally:
    from dotenv import load_dotenv


###########################################################
# Utility Functions
###########################################################

import os
import sys
from pathlib import Path

def detect_jupyter_environment():
    """
    Detects the Jupyter environment and returns one of:
    - "colab": Running in official Google Colab
    - "vscode": Running in VSCode
    - "cocalc": Running inside the CoCalc frontend
    - "cocalc_compute_server": Running on a compute server (e.g., GCP or Hyperstack) launched from CoCalc
    - "paperspace": Running in a Paperspace notebook
    - "unknown": Environment not recognized
    """

    # Check for official Google Colab
    if 'google.colab' in sys.modules:
        if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ:
            return "colab"

    # Check for VSCode
    if 'VSCODE_PID' in os.environ:
        return "vscode"

    # Check for Paperspace
    if 'PAPERSPACE_NOTEBOOK_ID' in os.environ:
        return "paperspace"

    # Check for CoCalc frontend (browser UI)
    if 'COCALC_CODE_PORT' in os.environ:
        return "cocalc"

    # Check for CoCalc Compute Server (GCP or Hyperstack)
    # CoCalc compute servers do NOT set COCALC_CODE_PORT, but do provision ~/cs_workspace
    if Path.home().joinpath("cs_workspace").exists():
        return "cocalc_compute_server"

    # Fallback
    return "unknown"


# def config_paths_keys(env_path="~/Lessons/Course_Tools/local.env", api_keys_env="~/Lessons/Course_Tools/api_keys.env"):
#     """
#     Reads environment variables and sets paths.
#     If variables are not set, it uses dotenv to load them based on the environment:
#     - CoCalc: ~/Lessons/Course_Tools/cocalc.env
#     - Colab: ~/Lessons/Course_Tools/colab.env
#     - Other: ~/Lessons/Course_Tools/local.env (default)

#     Additionally, loads API keys from api_keys_env if HF_TOKEN and OPENAI_API_KEY are not already set.

#     Parameters:
#         env_path (str): Path to the local environment file, defaulting to ~/Lessons/Course_Tools/local.env.
#         api_keys_env (str): Path to the API keys environment file, defaulting to ~/Lessons/Course_Tools/api_keys.env.

#     Returns:
#         dict: A dictionary with keys 'MODELS_PATH' and 'DATA_PATH'.
#     """

#     # Determine the environment
#     environment = detect_jupyter_environment()

#     # First, check if ~/local.env exists and use it if available
#     home_local_env = Path.home() / "local.env"

#     if home_local_env.exists():
#         env_file = home_local_env
#     else:
#         # If env_path is provided explicitly, use it
#         if env_path is not None:
#             env_file = Path(env_path).expanduser()
#         else:
#             # Choose a default env file based on the environment
#             if environment == "cocalc_compute_server":
#                 env_file = Path("~/Lessons/Course_Tools/cocalc_compute_server.env").expanduser()
#             elif environment == "cocalc":
#                 env_file = Path("~/Lessons/Course_Tools/cocalc.env").expanduser()
#             elif environment == "colab":
#                 env_file = Path("~/Lessons/Course_Tools/google_colab.env").expanduser()
#             else:  # fallback
#                 env_file = Path("~/Lessons/Course_Tools/local.env").expanduser()

#     # Load the environment variables
#     if env_file.exists():
#         load_dotenv(env_file, override=False)
#         print(f"Loaded environment variables from: {env_file}")
#     else:
#         print(f"Warning: environment file not found at {env_file}.  Pass path to env_path to load a different file.")

#     # Resolve path to ~/api_keys.env
#     home_api_keys_file = Path.home() / "api_keys.env"

#     # Determine which file to load
#     if home_api_keys_file.exists():
#         api_keys_file = home_api_keys_file
#     else:
#         if api_keys_env is not None:
#             api_keys_file = Path(api_keys_env).expanduser()
#         else:
#             if environment != "colab":
#                 api_keys_file = Path("~/Lessons/Course_Tools/api_keys.env").expanduser()
#             else:
#                 api_keys_file = Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env")

#     # Load the API keys from the selected file
#     if api_keys_file.exists():
#         load_dotenv(api_keys_file, override=False)
#         print(f"Loaded API keys from: {api_keys_file}")
#     else:
#         print(f"Warning: API keys file not found at {api_keys_file}. Pass path to api_keys_env to load a different file.")

#     # Retrieve and expand paths
#     models_path = Path(os.getenv('MODELS_PATH', "")).expanduser()
#     data_path = Path(os.getenv('DATA_PATH', "")).expanduser()
#     cache_path = Path(os.getenv('CACHE_PATH', "")).expanduser()
#     torch_home = Path(os.getenv('TORCH_HOME', "")).expanduser()
#     hf_home = Path(os.getenv('HF_HOME', "")).expanduser()

#     # Set environment variables to expanded paths
#     os.environ['MODELS_PATH'] = str(models_path)
#     os.environ['DATA_PATH'] = str(data_path)
#     os.environ['CACHE_PATH'] = str(cache_path)
#     os.environ['TORCH_HOME'] = str(cache_path)
#     os.environ['HF_HOME'] = str(cache_path)
#     os.environ['HF_DATASETS_CACHE'] = str(data_path)

#     # Create directories if they don't exist
#     for path in [models_path, data_path, cache_path, torch_home, hf_home]:
#         if not path.exists():
#             path.mkdir(parents=True, exist_ok=True)

#     # Ensure paths are set
#     print(f"MODELS_PATH={models_path}")
#     print(f"DATA_PATH={data_path}")
#     print(f"CACHE_PATH={cache_path}")
#     print(f"TORCH_HOME={torch_home}")
#     print(f"HF_HOME={hf_home}")
#     print(f"HF_DATASETS_CACHE={os.getenv('HF_DATASETS_CACHE')}")

#     # Login to Hugging Face if token is set
#     if os.getenv('HF_TOKEN'):
#         try:
#             import logging
#             logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
#             from huggingface_hub import login
#             login(token=os.getenv('HF_TOKEN'))
#             print("Successfully logged in to Hugging Face Hub.")
#         except Exception as e:
#             print(f"Failed to login to Hugging Face Hub: {e}")
#     else:
#         print("Set HF_TOKEN in api_keys.env or in environment to login to HuggingFace Hub")
#         print("Most things should work without logging in, but some features may be limited.")
    
#     return {
#         'MODELS_PATH': models_path,
#         'DATA_PATH': data_path,
#         'CACHE_PATH': cache_path
#     }

# def config_paths_keys(env_path="~/Lessons/Course_Tools/local.env", api_keys_env="~/Lessons/Course_Tools/api_keys.env"):
#     """
#     Reads environment variables and sets paths.

#     If running in Colab, sets hardcoded /content/temp_workspace paths.
#     Otherwise uses dotenv to load based on environment:
#     - CoCalc: ~/Lessons/Course_Tools/cocalc.env
#     - Local: ~/Lessons/Course_Tools/local.env

#     Also loads API keys from api_keys_env if HF_TOKEN or OPENAI_API_KEY are not already set.

#     Returns:
#         dict: A dictionary with keys 'MODELS_PATH', 'DATA_PATH', and 'CACHE_PATH'.
#     """

#     from pathlib import Path
#     import os
#     from dotenv import load_dotenv

#     environment = detect_jupyter_environment()

#     if environment == "colab":
#         # Set Colab-specific paths
#         base_path = Path("/content/temp_workspace")
#         data_path = base_path / "data"
#         models_path = base_path / "models"
#         cache_path = base_path / "downloads"

#         # Set environment variables
#         os.environ['DATA_PATH'] = str(data_path)
#         os.environ['MODELS_PATH'] = str(models_path)
#         os.environ['CACHE_PATH'] = str(cache_path)
#         os.environ['TORCH_HOME'] = str(cache_path)
#         os.environ['HF_HOME'] = str(cache_path)
#         os.environ['HF_DATASETS_CACHE'] = str(data_path)
#         os.environ['TQDM_NOTEBOOK'] = "true"

#         # Create the directories
#         for path in [data_path, models_path, cache_path]:
#             path.mkdir(parents=True, exist_ok=True)

#         print("[INFO] Environment: colab")
#         print(f"DATA_PATH={data_path}")
#         print(f"MODELS_PATH={models_path}")
#         print(f"CACHE_PATH={cache_path}")
#     else:
#         # Load local.env or appropriate default
#         home_local_env = Path.home() / "local.env"
#         if home_local_env.exists():
#             env_file = home_local_env
#         else:
#             env_file = Path(env_path).expanduser()

#         if env_file.exists():
#             load_dotenv(env_file, override=False)
#             print(f"Loaded environment variables from: {env_file}")
#         else:
#             print(f"Warning: environment file not found at {env_file}")

#         # Load API keys
#         home_api_keys_file = Path.home() / "api_keys.env"
#         if home_api_keys_file.exists():
#             api_keys_file = home_api_keys_file
#         elif environment == "colab":
#             api_keys_file = Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env")
#         else:
#             api_keys_file = Path(api_keys_env).expanduser()

#         if api_keys_file.exists():
#             load_dotenv(api_keys_file, override=False)
#             print(f"Loaded API keys from: {api_keys_file}")
#         else:
#             print(f"Warning: API keys file not found at {api_keys_file}")

#         # Retrieve and set paths
#         models_path = Path(os.getenv("MODELS_PATH", "")).expanduser()
#         data_path = Path(os.getenv("DATA_PATH", "")).expanduser()
#         cache_path = Path(os.getenv("CACHE_PATH", "")).expanduser()

#         os.environ["TORCH_HOME"] = str(cache_path)
#         os.environ["HF_HOME"] = str(cache_path)
#         os.environ["HF_DATASETS_CACHE"] = str(data_path)

#         for path in [models_path, data_path, cache_path]:
#             if not path.exists():
#                 path.mkdir(parents=True, exist_ok=True)

#         print(f"MODELS_PATH={models_path}")
#         print(f"DATA_PATH={data_path}")
#         print(f"CACHE_PATH={cache_path}")

#     # Hugging Face login if token exists
#     if os.getenv("HF_TOKEN"):
#         try:
#             import logging
#             logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
#             from huggingface_hub import login
#             login(token=os.getenv("HF_TOKEN"))
#             print("Successfully logged in to Hugging Face Hub.")
#         except Exception as e:
#             print(f"Failed to login to Hugging Face Hub: {e}")
#     else:
#         print("Set HF_TOKEN in api_keys.env or the environment to login to Hugging Face Hub")

#     return {
#         'MODELS_PATH': models_path,
#         'DATA_PATH': data_path,
#         'CACHE_PATH': cache_path
#     }

def config_paths_keys(env_path="~/Lessons/Course_Tools/local.env", api_keys_env="~/Lessons/Course_Tools/api_keys.env"):
    """
    Reads environment variables and sets paths.

    If running in Colab, sets hardcoded /content/temp_workspace paths.
    Otherwise uses dotenv to load based on environment:
    - CoCalc: ~/Lessons/Course_Tools/cocalc.env
    - Local: ~/Lessons/Course_Tools/local.env

    Also loads API keys from api_keys.env if HF_TOKEN or OPENAI_API_KEY are not already set.

    Returns:
        dict: A dictionary with keys 'MODELS_PATH', 'DATA_PATH', and 'CACHE_PATH'.
    """

    environment = detect_jupyter_environment()

    if environment == "colab":
        # Set Colab-specific paths
        base_path = Path("/content/temp_workspace")
        data_path = base_path / "data"
        models_path = base_path / "models"
        cache_path = base_path / "downloads"

        # Set environment variables
        os.environ['DATA_PATH'] = str(data_path)
        os.environ['MODELS_PATH'] = str(models_path)
        os.environ['CACHE_PATH'] = str(cache_path)
        os.environ['TORCH_HOME'] = str(cache_path)
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['HF_DATASETS_CACHE'] = str(data_path)
        os.environ['TQDM_NOTEBOOK'] = "true"

        # Create the directories
        for path in [data_path, models_path, cache_path]:
            path.mkdir(parents=True, exist_ok=True)

        print("[INFO] Environment: colab")
        print(f"DATA_PATH={data_path}")
        print(f"MODELS_PATH={models_path}")
        print(f"CACHE_PATH={cache_path}")

    else:
        # Load local.env or environment-specific default
        home_local_env = Path.home() / "local.env"
        if home_local_env.exists():
            env_file = home_local_env
        else:
            env_file = Path(env_path).expanduser()

            if not env_file.exists():
                # Auto-choose based on environment
                if environment == "cocalc_compute_server":
                    env_file = Path("~/Lessons/Course_Tools/cocalc_compute_server.env").expanduser()
                elif environment == "cocalc":
                    env_file = Path("~/Lessons/Course_Tools/cocalc.env").expanduser()
                elif environment == "colab":
                    env_file = Path("~/Lessons/Course_Tools/google_colab.env").expanduser()
                else:
                    env_file = Path("~/Lessons/Course_Tools/local.env").expanduser()

        if env_file.exists():
            load_dotenv(env_file, override=False)
            print(f"Loaded environment variables from: {env_file}")
        else:
            print(f"Warning: environment file not found at {env_file}")

        # Retrieve and set paths
        models_path = Path(os.getenv("MODELS_PATH", "")).expanduser()
        data_path = Path(os.getenv("DATA_PATH", "")).expanduser()
        cache_path = Path(os.getenv("CACHE_PATH", "")).expanduser()

        os.environ["TORCH_HOME"] = str(cache_path)
        os.environ["HF_HOME"] = str(cache_path)
        os.environ["HF_DATASETS_CACHE"] = str(data_path)

        for path in [models_path, data_path, cache_path]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

        print(f"MODELS_PATH={models_path}")
        print(f"DATA_PATH={data_path}")
        print(f"CACHE_PATH={cache_path}")

    # 🔐 Load API keys (colab-aware)
    api_keys_file = None
    home_api_keys_file = Path.home() / "api_keys.env"
    colab_api_keys_file = Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env")

    if home_api_keys_file.exists():
        api_keys_file = home_api_keys_file
    elif environment == "colab" and colab_api_keys_file.exists():
        api_keys_file = colab_api_keys_file
    elif api_keys_env:
        api_keys_file = Path(api_keys_env).expanduser()

    if api_keys_file and api_keys_file.exists():
        load_dotenv(api_keys_file, override=False)
        print(f"Loaded API keys from: {api_keys_file}")
    else:
        print(f"Warning: API keys file not found. Looked in {home_api_keys_file} and {colab_api_keys_file}")

    # 🔐 Login to Hugging Face
    if os.getenv("HF_TOKEN"):
        try:
            import logging
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            from huggingface_hub import login
            login(token=os.getenv("HF_TOKEN"))
            print("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"Failed to login to Hugging Face Hub: {e}")
    else:
        print("Set HF_TOKEN in api_keys.env or in the environment to login to Hugging Face Hub")

    return {
        'MODELS_PATH': models_path,
        'DATA_PATH': data_path,
        'CACHE_PATH': cache_path
    }


def get_device():
    """
    Returns the appropriate device ('cuda', 'mps', or 'cpu') depending on availability.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def cleanup_torch(*objects):
    """Delete objects, clear CUDA cache, and run garbage collection."""
    for obj in objects:
        try:
            del obj
        except:
            pass
    torch.cuda.empty_cache()
    gc.collect()


def hf_download(checkpoint_file, repo_id, token=None):
    """
    Download a file directly from the Hugging Face repository.

    Parameters:
    - checkpoint_file (str): The path to the local file where the downloaded file will be saved.
    - repo_id (str): Hugging Face repository ID.
    - token (str, optional): Hugging Face access token for private repositories.

    Returns:
    - None: The file is saved directly to the checkpoint_file location.
    """
    import os
    import requests

    # Construct the file download URL
    base_url = "https://huggingface.co"
    filename = os.path.basename(checkpoint_file)
    file_url = f"{base_url}/{repo_id}/resolve/main/{filename}"

    # Download the file directly
    response = requests.get(file_url, stream=True, headers={"Authorization": f"Bearer {token}"} if token else {})
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to download '{filename}' from {file_url}. Status code: {response.status_code}")

    # Write the file to the desired checkpoint_file location
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def load_results(checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", token=None):
    """
    Load the results from a checkpoint file.

    Parameters:
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - token (str, optional): Hugging Face access token for private repositories.

    Returns:
    - results (pd.DataFrame): The loaded results from the checkpoint file.
    """

    # Download the file if it does not exist locally
    if not os.path.exists(checkpoint_file):
        if repo_id is None:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found locally, and no repo_id provided.")
        hf_download(checkpoint_file, repo_id, token)

    # Suppress FutureWarning during torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

    # Extract the results
    if 'results' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'results'.")
    return pd.DataFrame(checkpoint_dict['results'])

def load_model(model, checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", token=None):
    """
    Load the model from a checkpoint file, trying locally first, then downloading if not found.

    Parameters:
    - model: The model to load. It can be either a class or an instance of the model.
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - token (str, optional): Hugging Face access token for private repositories.

    Returns:
    - model: The loaded model from the checkpoint file.
    """

    # Download the file if it does not exist locally
    if not os.path.exists(checkpoint_file):
        if repo_id is None:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found locally, and no repo_id provided.")
        hf_download(checkpoint_file, repo_id, token)

    # Instantiate model if a class is passed
    if inspect.isclass(model):
        model = model()
    elif not isinstance(model, nn.Module):
        raise ValueError("The model must be a class or an instance of nn.Module.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

    if 'model_state_dict' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    return model.to(device)


def summarizer(model, input_size, device=torch.device('cpu'), col_width=20, verbose=False, varnames = True, **kwargs):
    """
    Summarizes the given model by displaying the input size, output size, and number of parameters.

    Parameters:
    - model: The model to summarize.
    - input_size (tuple): The input size of the model.
    - device (torch.device, optional): The device to summarize the model on. Defaults to 'cpu'.
    - col_width (int, optional): The width of each column in the summary table. Defaults to 20.
    - verbose (bool, optional): If True, display the full error stack trace; otherwise, show only a simplified error message. Defaults to False.
    - **kwargs: Additional keyword arguments to pass to the summary function.
    """
    model = model.to(device)
    try:
        colnames = ["input_size", "output_size", "num_params"]
        rowsettings = ["var_names"] if varnames else ["depth"]
        print(summary(model, input_size=input_size, col_width=col_width, row_settings=rowsettings, col_names=colnames, **kwargs))
    except RuntimeError as e:
        if verbose:
            # Print the full stack trace and original error message
            traceback.print_exc()
            print(f"Original Error: {e}")
        else:
            # Display simplified error message with additional message for verbose option
            error_message = str(e).splitlines()[-1].replace("See above stack traces for more details.", "").strip()
            error_message = error_message.replace("Failed to run torchinfo.", "Failed to run all model layers.")
            error_message += " Run again with verbose=True to see stack trace."
            print(f"Error: {error_message}")

def classifier_predict(dataset, model, device, batch_size=32, return_labels=False):
    """
    Collects predictions from a PyTorch dataset using a classification model.
    Optionally returns ground truth labels.

    Assumptions:
        - The model outputs logits for each class (not probabilities or class indices).
        - The dataset returns tuples of (inputs, labels) where labels are integers representing class indices.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to evaluate.
        model (torch.nn.Module): The classification model. Assumes outputs are logits for each class.
        device (torch.device): The device to run the evaluation on.
        return_labels (bool): Whether to return ground truth labels along with predictions.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: Predicted labels (class indices).
        list (optional): Ground truth labels (if return_labels=True).
    """
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # Initialize lists to store predictions and ground truth labels
    predictions = []
    ground_truth = [] if return_labels else None

    # Turn off gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device
            inputs = inputs.to(device)
            if return_labels:
                labels = labels.to(device)

            # Forward pass through the model
            logits = model(inputs)

            # Get predicted labels (the class with the highest logit)
            preds = torch.argmax(logits, dim=1)

            # Append predictions to the list
            predictions.extend(preds.cpu().tolist())
            # Append ground truth labels if requested
            if return_labels:
                ground_truth.extend(labels.cpu().tolist())

    if return_labels:
        return predictions, ground_truth
    return predictions

def create_CIFAR10_loaders(transform_train=None, transform_test=None, transform_valid=None,
                           valid_prop=0.2, batch_size=64, seed=42, data_dir='./data', 
                           downsample_prop=1.0, num_workers=1, persistent_workers = True, 
                           use_augmentation=False):
    """
    Create data loaders for the CIFAR10 dataset.

    Args:
        transform_train (torchvision.transforms.v2.Compose, optional): Transformations for the training set. Defaults to standard training transforms if None.
        transform_test (torchvision.transforms.v2.Compose, optional): Transformations for the test set. Defaults to standard test transforms if None.
        transform_valid (torchvision.transforms.v2.Compose, optional): Transformations for the validation set. Defaults to None.
        valid_prop (float or None): Proportion of the training set to use for validation. If 0.0 or None, no validation split is made.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to download/load CIFAR10 data.
        downsample_prop (float): Proportion of the dataset to keep if less than 1. Defaults to 1.0.
        num_workers (int): Number of worker processes to use for data loading.
        use_augmentation (bool): Whether to apply data augmentation to the training set. Defaults to False.

    Returns:
        tuple: Train loader, test loader, and optionally valid loader, along with the datasets.
    """

    # Set default transforms if none are supplied
    mean = (0.4914, 0.4822, 0.4465) 
    std = (0.2023, 0.1994, 0.2010)

    if transform_train is None:
        if use_augmentation:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=mean, std=std), 
                transforms.ToPureTensor()   
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=mean, std=std),
                transforms.ToPureTensor()   
            ])
    
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToPureTensor()   
        ])
        
    # Set validation transform; if None, use transform_test
    if transform_valid is None:
        transform_valid = transform_test

    # Load the full training and test datasets
    train_dataset_full = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Generate indices for training and validation if needed
    train_indices, valid_indices = None, None
    if valid_prop and 0 < valid_prop < 1.0:
        total_indices = list(range(len(train_dataset_full)))
        train_indices, valid_indices = train_test_split(
            total_indices,
            test_size=valid_prop,
            random_state=seed,
            shuffle=True
        )

    # Downsample datasets if required
    if downsample_prop < 1.0:
        train_indices = train_indices[:int(downsample_prop * len(train_indices))] if train_indices else None
        valid_indices = valid_indices[:int(downsample_prop * len(valid_indices))] if valid_indices else None

    # Create Subset datasets for training and optionally validation
    train_dataset = Subset(train_dataset_full, train_indices) if train_indices else train_dataset_full
    valid_dataset = Subset(CIFAR10(root=data_dir, train=True, download=True, transform=transform_valid), valid_indices) if valid_indices else None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, persistent_workers=persistent_workers)
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=num_workers, persistent_workers=persistent_workers)

    if valid_loader:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader

# def wrap_print_text(print):
#     """
#     Wraps the given print function to format text with a specified width.
#     This function takes a print function as an argument and returns a new function
#     that formats the text to a specified width before printing. The text is wrapped
#     to 80 characters per line, and long words are broken to fit within the width.
#     Args:
#         print (function): The original print function to be wrapped.
#     Returns:
#         function: A new function that formats text to 80 characters per line and
#                   then prints it using the original print function.
#     Example:
#         wrapped_print = wrap_print_text(print)
#         wrapped_print("This is a very long text that will be wrapped to fit within 80 characters per line.")
#     Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

#     def wrapped_func(text):
#         if not isinstance(text, str):
#             text = str(text)
#         wrapper = TextWrapper(
#             width=80,
#             break_long_words=True,
#             break_on_hyphens=False,
#             replace_whitespace=False,
#         )
#         return print("\n".join(wrapper.fill(line) for line in text.split("\n")))

#     return wrapped_func

def wrap_print_text(original_print, width=80):
    """
    Wraps the given print function to format text with a specified width.
    This function takes a print function as an argument and returns a new function
    that formats the text to a specified width before printing. The text is wrapped
    to the specified number of characters per line, and long words are broken to fit.
    """

    def wrapped_func(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        wrapper = TextWrapper(
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        wrapped_text = "\n".join(wrapper.fill(line) for line in text.split("\n"))
        return original_print(wrapped_text, **kwargs)

    return wrapped_func

def _guess_notebook_path():
    """
    Guess the current notebook path by looking for the most recently modified .ipynb file in the working dir.
    """
    cwd = Path.cwd()
    candidates = sorted(cwd.glob("*.ipynb"), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise RuntimeError("No .ipynb files found in the current directory.")
    print(f"[INFO] Using notebook: {candidates[0].name}")
    return candidates[0]

def _clean_invalid_outputs(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        new_outputs = []
        for j, output in enumerate(cell.get("outputs", [])):
            fake_nb = {
                "cells": [{
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": 1,
                    "outputs": [output],
                    "source": "",
                    "id": f"cell-{i}-{j}"
                }],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5
            }
            try:
                normalize(fake_nb)
                validate(fake_nb, relax_add_props=True)
                new_outputs.append(output)
            except NotebookValidationError:
                print(f"[WARN] Removed invalid output from cell {i}, output {j}")

        cell["outputs"] = new_outputs

    normalize(nb)

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def convert_nb_to_html(output_filename="converted.html", notebook_path=None):
    """
    Convert a notebook to HTML using the JupyterLab template.
    If notebook_path is None, uses the most recent .ipynb file in the current directory.
    Output will be written to current directory with the given output_filename.
    """
    output_filename = str(output_filename)
    if not output_filename.endswith(".html"):
        output_filename += ".html"

    if notebook_path is None:
        notebook_path = _guess_notebook_path()

    notebook_path = Path(notebook_path).resolve()
    output_path = Path.cwd() / output_filename

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / notebook_path.name
        shutil.copy2(notebook_path, tmp_path)
        print(f"[INFO] Temporary copy created: {tmp_path}")

        _clean_invalid_outputs(tmp_path)

        try:
            subprocess.run(
                [
                    "jupyter", "nbconvert",
                    "--to", "html",
                    "--template", "lab",
                    "--output", output_path.stem,
                    "--output-dir", str(output_path.parent),
                    str(tmp_path)
                ],
                check=True
            )
            print(f"[SUCCESS] HTML export complete: {output_path}")
        except subprocess.CalledProcessError as e:
            print("[ERROR] Conversion failed:", e)
