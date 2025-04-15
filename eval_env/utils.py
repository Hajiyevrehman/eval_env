########################
# Utils Functions
########################

import multiprocessing
import subprocess
import re
import random
import tempfile
from pathlib import Path
import re
import math
import os
import json
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def construct_problem_dataset_from_problem_dir(problem_dir):
    # Dummy implementation for compatibility
    # In your real code, replace this with actual logic
    return {0: os.path.join(problem_dir, "dummy_problem.py")}
