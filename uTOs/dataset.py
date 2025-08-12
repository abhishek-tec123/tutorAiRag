# First, you need to install the datasets library if you haven't already.
# You can can do this by running: pip install datasets

import json
from datasets import load_dataset

# Step 1: Load the 'lukalafaye/IMDB-Metadata' dataset.
# This dataset contains much more detail about the movies.
print("Loading the 'lukalafaye/IMDB-Metadata' dataset...")
try:
    # We load the entire 'train' split as it's a smaller, consolidated dataset.
    imdb_metadata = load_dataset("lukalafaye/IMDB-Metadata", split="train")
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    imdb_metadata = None

if imdb_metadata:
    # Step 2: Convert a small portion of the dataset to a Python list of dictionaries.
    # We will take the first 50 entries to create a manageable JSON file.
    data_list = imdb_metadata.to_list()[:50]

    # Step 3: Define the output filename.
    output_filename = "imdb_full_details.json"
    
    # Step 4: Write the list of dictionaries to a JSON file.
    # The 'indent' parameter makes the JSON output easy to read.
    with open(output_filename, 'w') as f:
        json.dump(data_list, f, indent=2)

    print(f"\nSuccessfully saved a subset of detailed movie data to {output_filename}")
    
    # Step 5: Print the first 5 entries to the console to showcase the new structure.
    print(f"\nHere are the first 5 entries from the data saved in '{output_filename}':")
    for i, entry in enumerate(data_list[:5]):
        print(f"--- Entry {i+1} ---")
        print(json.dumps(entry, indent=2))
        print("--------------------")

