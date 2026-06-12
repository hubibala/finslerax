"""Utility script for downloading benchmark datasets.

Downloads and preprocesses the scVelo Pancreas dataset, computing velocity
estimates to serve as a ground-truth vector field for testing.
"""

import logging

import scvelo as scv

logger = logging.getLogger(__name__)


def download_benchmark_data() -> str:
    """Download and preprocess the scVelo Pancreas dataset.

    Returns:
        str: Path to the processed .h5ad file.
    """
    logger.info("Downloading Pancreas data (approx 30MB)...")
    # This automatically saves 'pancreas.h5ad' to ./data/
    adata_pancreas = scv.datasets.pancreas()

    # Pre-calculate velocity vectors so we have a "Ground Truth" V to train on
    logger.info("Preprocessing & estimating velocity...")
    scv.pp.filter_and_normalize(adata_pancreas)
    scv.pp.moments(adata_pancreas)
    scv.tl.velocity(adata_pancreas, mode="stochastic")
    scv.tl.velocity_graph(adata_pancreas)

    # Save the processed file
    filename = "data/pancreas_processed.h5ad"
    adata_pancreas.write(filename)
    logger.info(f"Saved to {filename}")
    return filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_benchmark_data()
