# FILE: export_reid_weights.py
import argparse
import time
import sys
import logging
from pathlib import Path
from collections import OrderedDict # <-- Added

import torch
# --- Add src to path to find project modules ---
# (Path setup code remains the same as previous version)
PROJECT_ROOT = Path(__file__).parent.resolve()
if PROJECT_ROOT.name == 'scripts': # Handle running from script/ or project root
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = PROJECT_ROOT / "src"
# ----> Adjust this path based on your actual venv location relative to PROJECT_ROOT <----
VENV_PATH = PROJECT_ROOT / ".venv"
BOXMOT_PATH = VENV_PATH / "Lib/site-packages/boxmot"
#------------------------------------------------------------------------------------

# Simple logger for the script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Add project paths
if str(SRC_PATH) not in sys.path: sys.path.insert(0, str(SRC_PATH))
if str(BOXMOT_PATH) not in sys.path:
    try: import boxmot; logger.info(f"Found boxmot at {boxmot.__path__}")
    except ImportError:
        logger.warning(f"boxmot not found directly, adding path: {BOXMOT_PATH}")
        if BOXMOT_PATH.exists(): sys.path.insert(0, str(BOXMOT_PATH))
        else: logger.error(f"BoxMOT path does not exist: {BOXMOT_PATH}. Cannot import.")
# --- End Path Setup ---

# --- Imports from BoxMOT (handle potential errors) ---
try:
    from boxmot.appearance.reid.registry import ReIDModelRegistry
    from boxmot.utils.torch_utils import select_device
    from boxmot.utils import logger as BOXMOT_LOGGER
except ImportError as e:
    logger.error(f"Error importing BoxMOT components. Is BoxMOT installed and path correct? Error: {e}")
    sys.exit(1)
# --- End BoxMOT Imports ---


def parse_args():
    """Parse command-line arguments for the ReID re-saving script."""
    # (Argument parsing remains the same)
    parser = argparse.ArgumentParser(description="ReID Weight Re-Saving Script")
    parser.add_argument(
        "--input-weights", type=Path, required=True,
        help="Path to the original/problematic ReID weights (.pt file)",
    )
    parser.add_argument(
        "--output-weights", type=Path, required=True,
        help="Path to save the new, potentially compatible weights (.pt file)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to attempt loading ONTO ('cpu', 'cuda:0', etc.)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Optional: Explicitly specify model type (e.g., 'mlfn'). If None, infer from filename.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Optional: Explicitly specify number of classes. If None, infer from filename.",
    )
    return parser.parse_args()

def load_state_dict_flexible(model: torch.nn.Module, checkpoint_state_dict: OrderedDict):
    """
    Loads state_dict, handling potential 'module.' prefix and mismatched keys gracefully.
    """
    # (Helper function remains the same)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_keys = []
    unmatched_keys = []
    mismatched_size_keys = []

    for k, v in checkpoint_state_dict.items():
        key_no_prefix = k[7:] if k.startswith("module.") else k
        if key_no_prefix in model_dict:
            if model_dict[key_no_prefix].size() == v.size():
                new_state_dict[key_no_prefix] = v
                matched_keys.append(key_no_prefix)
            else:
                mismatched_size_keys.append(key_no_prefix)
                unmatched_keys.append(k) # Add original key
        else:
            unmatched_keys.append(k) # Add original key

    if not new_state_dict:
         raise RuntimeError(f"Failed to load state_dict. No keys matched between checkpoint and model architecture.")

    model_dict.update(new_state_dict)
    try:
        model.load_state_dict(model_dict)
        logger.info(f"Successfully loaded state_dict with {len(matched_keys)} matched keys into model.")
        if mismatched_size_keys:
             logger.warning(f"Skipped loading {len(mismatched_size_keys)} keys due to size mismatch: {mismatched_size_keys}")
        if unmatched_keys:
             log_limit = 10
             logger.warning(f"Found {len(unmatched_keys)} unmatched keys in checkpoint (examples: {unmatched_keys[:log_limit]}{'...' if len(unmatched_keys) > log_limit else ''}). These were not loaded.")

    except Exception as e:
         logger.error(f"Error during model.load_state_dict: {e}")
         raise e


def main():
    """Main function to load and re-save ReID weights."""
    args = parse_args()
    start_time = time.time()

    if not args.input_weights.is_file():
        logger.error(f"Input weights file not found: {args.input_weights}")
        sys.exit(1)

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Attempting to re-save weights:")
    logger.info(f"  Input:  {args.input_weights}")
    logger.info(f"  Output: {args.output_weights}")
    logger.info(f"  Target Device: {args.device}")

    try:
        # --- 1. Select TARGET Device ---
        target_device = select_device(args.device)
        logger.info(f"Target device selected: {target_device}")

        # --- 2. Determine Model Name and Number of Classes ---
        # (Logic remains the same)
        model_name = args.model_name or ReIDModelRegistry.get_model_name(args.input_weights)
        if not model_name:
             logger.error(f"Could not infer model name from filename: {args.input_weights.name}")
             logger.error("Please specify the model name using --model-name (e.g., 'mlfn', 'resnet50')")
             sys.exit(1)
        num_classes = args.num_classes or ReIDModelRegistry.get_nr_classes(args.input_weights)
        if not num_classes:
             logger.warning(f"Could not infer number of classes from filename: {args.input_weights.name}. Using default: 1")
             num_classes = 1
        logger.info(f"Inferred/Using Model Name: {model_name}, Num Classes: {num_classes}")

        # --- 3. Build Model Architecture ---
        # (Logic remains the same)
        logger.info(f"Building model architecture: {model_name} on {target_device}...")
        model = ReIDModelRegistry.build_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=False,
            use_gpu=target_device
        ).to(target_device)
        model.eval()
        logger.info("Model architecture built successfully.")

        # --- 4. Load Original Weights DIRECTLY with map_location AND weights_only=False ---
        # *** MODIFIED LOADING ***
        logger.info(f"Loading checkpoint from: {args.input_weights}...")
        # Explicitly set map_location AND weights_only=False
        # This allows loading older files with pickled non-tensor data
        try:
            checkpoint = torch.load(
                args.input_weights,
                map_location=target_device,
                weights_only=False  # <-- Key change: Allow loading older, less safe formats
            )
            logger.info("Checkpoint loaded successfully (using weights_only=False).")
        except AttributeError as ae:
             # Catch potential errors if weights_only=False leads to unexpected object types
             logger.error(f"Loading Error (AttributeError): {ae}")
             logger.error("The loaded checkpoint structure might be unexpected after using weights_only=False.")
             sys.exit(1)


        # Extract the state dictionary (handle potential nesting or direct dict)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, (OrderedDict, dict)):
             logger.error(f"Loaded checkpoint does not contain a valid state dictionary (found type: {type(state_dict)}). Cannot proceed.")
             sys.exit(1)
        # Ensure state_dict is OrderedDict for load_state_dict_flexible
        if not isinstance(state_dict, OrderedDict):
            state_dict = OrderedDict(state_dict)


        # Load the state dict into the model structure flexibly
        logger.info("Loading state dict into model architecture...")
        load_state_dict_flexible(model, state_dict)
        # *** END MODIFIED LOADING ***

        # --- 5. Re-Save the State Dictionary ---
        logger.info(f"Re-saving model state_dict to: {args.output_weights}...")
        # Save the state_dict of the model (which is now on the target_device)
        # This save operation uses the current PyTorch version's default (likely safe) protocol
        torch.save(model.state_dict(), args.output_weights)
        logger.success("Model state_dict re-saved successfully!")

        elapsed_time = time.time() - start_time
        logger.info(f"\nRe-saving complete ({elapsed_time:.1f}s)")
        logger.info(f"New weights file saved to: {args.output_weights.resolve()}")
        logger.info("You can now try using this new weights file in your configuration.")

    except ValueError as ve:
        # Catch the original numpy encoding error if it persists even with weights_only=False
        logger.error(f"Loading Error (ValueError): {ve}")
        logger.error("This likely indicates the input .pt file's pickled NumPy format is incompatible even when allowing unsafe loading.")
        logger.error("Re-saving failed. The original file cannot be loaded in this environment.")
        sys.exit(1)
    except UnicodeDecodeError as ude:
        logger.error(f"Loading Error (UnicodeDecodeError): {ude}")
        logger.error("This likely indicates the input .pt file is corrupted or not a standard PyTorch file.")
        logger.error("Re-saving failed. The original file cannot be loaded.")
        sys.exit(1)
    except FileNotFoundError as fnf:
        logger.error(f"File Not Found Error: {fnf}")
        sys.exit(1)
    except RuntimeError as rte: # Catch RuntimeErrors like CUDA mapping error
         if "Attempting to deserialize object on CUDA device" in str(rte):
              logger.error(f"Loading Error (RuntimeError): {rte}")
              logger.error("This confirms the weights were saved on CUDA but it's unavailable or not targeted.")
              logger.error("Ensure --device cpu is used if you want to load to CPU, or ensure CUDA is available and targeted if loading to GPU.")
         else:
              logger.error(f"An unexpected RuntimeError occurred: {rte}", exc_info=True)
         logger.error("Re-saving failed.")
         sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        logger.error("Re-saving failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()