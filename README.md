# Near-Duplicate Image Detection Using CLIP

This project uses OpenAI's CLIP image encoder to find visually similar and near-duplicate images. It converts images into normalized feature vectors (embeddings) and compares them with cosine similarity.

[Open the dataset on Google Drive](https://drive.google.com/drive/folders/10N3kP_oHp9uznpC7uQjEMhNFO7Urn3up?usp=drive_link)

## Features

- **Build an image database:** Generate and store an embedding for each supported image in a folder.
- **Find near-duplicates:** Compare images in a query folder against an existing embedding database and report each query image's best matching database image when its similarity reaches a threshold.
- **Remove duplicates:** Compare every pair of images in one folder and delete the second image in a pair when their similarity reaches the configured threshold.
- **Search for similar but diverse images:** Rank database images by similarity to a query image, then skip results that are too similar to an already selected result.
- **Store only unique images:** Add an image to an in-memory database only if it is not too similar to an image already stored.
- **Evaluate matches:** Calculate precision, recall, and F1 score from predicted and ground-truth match sets.

## How it works

1. CLIP's `ViT-B/32` model preprocesses each image and creates an image embedding.
2. The embedding is normalized so that cosine similarity can compare images consistently.
3. Similarity is calculated between query and stored embeddings. Scores closer to `1` indicate greater similarity.
4. A threshold determines whether a match is considered a near-duplicate. The threshold can be adjusted for the dataset and use case.

The model runs on a CUDA-capable GPU when available; otherwise, it uses the CPU.

## Requirements

- Python 3
- PyTorch
- OpenAI CLIP
- NumPy
- Pillow
- scikit-learn
- tqdm

Install the Python packages and CLIP with:

```bash
pip install torch torchvision numpy pillow scikit-learn tqdm
pip install git+https://github.com/openai/CLIP.git
```

For GPU support, install the PyTorch build that matches your CUDA setup by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Dataset layout

The current script expects these paths relative to the directory from which it is run:

```text
datasets/
├── copydays/
│   ├── original/       # Images used to build the comparison database
│   ├── transformed/    # Transformed images (currently embedded by the script)
│   └── test/           # Query images checked for duplicates
└── landmarks/
    └── test.jpg        # Example image used for search and unique-image checks
```

Update the paths near the bottom of the Python file if your folders are arranged differently. The dataset link above points to the dataset; download it and place the files into the expected layout before running the script.

## Running the project

Run the Python file from the project root after installing the dependencies and setting up the dataset paths:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the name of your Python file. The current file runs example operations at the bottom: it builds embeddings for `datasets/copydays/original` and `datasets/copydays/transformed`, searches for diverse images using `datasets/landmarks/test.jpg`, checks whether that image is unique, searches `datasets/copydays/test` for duplicates, and computes evaluation metrics.

The functions can also be called from other Python code. For example:

```python
from your_script_name import build_database, find_duplicates

database = build_database("datasets/copydays/original")
matches = find_duplicates(
    "datasets/copydays/test",
    database,
    threshold=0.75,
)

for query_image, matched_images in matches.items():
    print(query_image, matched_images)
```

**Note:** The current script runs the example code at the bottom as soon as the file is executed or imported. To use the functions as an importable module without running those examples, move the example section into a `main()` function and call it under `if __name__ == "__main__":`.

## Main functions

| Function | Purpose |
|---|---|
| `get_embedding(image_path)` | Loads and preprocesses one image, then returns its normalized CLIP embedding. Returns `None` if processing fails. |
| `build_database(image_folder)` | Creates a dictionary mapping image paths to embeddings for supported images in a folder. |
| `find_duplicates(query_folder, embeddings_db, threshold=0.93)` | Finds the best database match for each query image if its similarity is at least the threshold. |
| `remove_duplicate_images(image_folder, threshold=0.95)` | Finds similar image pairs and deletes marked images from the folder. |
| `search_different(query_image, embeddings_db, top_k=5, diversity_threshold=0.85)` | Returns up to `top_k` high-similarity results while filtering results that are too similar to one another. |
| `store_if_unique(image_path, embeddings_db, threshold=0.95)` | Adds an image embedding to the database only if no existing embedding reaches the duplicate threshold. |
| `compute_f1(predicted, ground_truth)` | Prints and returns precision, recall, and F1 score for the predicted and ground-truth match dictionaries. |

Supported image extensions are `.jpg`, `.jpeg`, and `.png`.

## Choosing a threshold

Thresholds are similarity cutoffs, not universal definitions of a duplicate. A higher threshold is stricter and generally requires images to be more alike; a lower threshold accepts more possible matches and may produce more false positives. Test different values on labeled examples and use `compute_f1` to help choose a suitable value. The script currently uses different thresholds in different examples, so set them deliberately for your dataset.

## Important safety note

`remove_duplicate_images()` permanently deletes files using `os.remove()`. It does not move them to a recycle bin or create a backup. Back up your images first, and test the function on a copy of your dataset before using it on important files.

## Current limitations

- Embeddings are held in memory and are not saved to disk, so the database must be rebuilt each run.
- `find_duplicates()` reports only the best match in the database for each query image.
- The duplicate-removal function uses pairwise comparisons, which become slow as the number of images grows.
- The script's example ground-truth dictionary is a small placeholder; replace it with labels that match your dataset before treating the reported metrics as meaningful.
- Only the listed image file extensions are included.

## License

Add a license file or license statement here if you plan to distribute or reuse this project.
