# ─── Standard library ────────────────────────────────────────────────────────
import io
import logging
import os
import time
import warnings
import zipfile
from pathlib import Path
from typing import Optional

# ─── Third-party ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ─── Suppress noisy deprecation warnings from third-party libs ───────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global Constants
# ─────────────────────────────────────────────────────────────────────────────

MOVIELENS_URL: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR: Path = Path("data/ml-100k")
PLOTS_DIR: Path = Path("plots")
RANDOM_SEED: int = 42
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib / Seaborn global style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor": "#161625",
        "axes.edgecolor": "#2a2a3d",
        "axes.labelcolor": "#c8c8e0",
        "xtick.color": "#8888aa",
        "ytick.color": "#8888aa",
        "text.color": "#c8c8e0",
        "grid.color": "#2a2a3d",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.facecolor": "#1e1e30",
        "legend.edgecolor": "#2a2a3d",
        "figure.dpi": 120,
    }
)

# Accent palette used across all charts
ACCENT_COLORS: list[str] = ["#7c6af7", "#f7b731", "#26de81", "#fc5c65", "#45aaf2"]

# ─────────────────────────────────────────────────────────────────────────────
# Data Acquisition & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def download_movielens(url: str = MOVIELENS_URL, dest: Path = DATA_DIR) -> Path:
    """Download and extract the MovieLens 100K dataset if not already present.

    The function checks for the two key data files (u.data, u.item) before
    attempting any network request. If they are already on disk the download
    is skipped entirely, allowing fully offline / pre-seeded runs.

    Args:
        url:  Direct URL to the MovieLens 100K zip archive.
        dest: Local directory where the dataset will be extracted.

    Returns:
        Path to the directory containing the raw data files.

    Raises:
        requests.HTTPError: If the HTTP download request fails.
        zipfile.BadZipFile: If the downloaded archive is corrupted.
        FileNotFoundError: If the data cannot be obtained from any source.
    """
    # Skip download if the essential files are already present on disk
    if (dest / "u.data").exists() and (dest / "u.item").exists():
        logger.info("Dataset already exists at '%s'. Skipping download.", dest)
        return dest

    logger.info("Downloading MovieLens 100K from %s …", url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise requests.HTTPError(f"Failed to download dataset: {exc}") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(dest.parent)
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile("Downloaded archive is not a valid ZIP file.") from exc

    logger.info("Dataset extracted to '%s'.", dest)
    return dest


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the u.data ratings file into a tidy DataFrame.

    Args:
        data_dir: Directory that contains the MovieLens raw files.

    Returns:
        DataFrame with columns [user_id, movie_id, rating, timestamp].

    Raises:
        FileNotFoundError: If the ratings file does not exist.
    """
    ratings_path = data_dir / "u.data"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found at '{ratings_path}'. "
            "Run download_movielens() first."
        )

    df = pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "movie_id": np.int32, "rating": np.float32},
    )

    # Validate expected value ranges
    assert df["rating"].between(1, 5).all(), "Rating values out of [1, 5] range."
    assert df["user_id"].gt(0).all(), "User IDs must be positive integers."
    assert df["movie_id"].gt(0).all(), "Movie IDs must be positive integers."

    logger.info(
        "Loaded %d ratings | %d users | %d movies",
        len(df),
        df["user_id"].nunique(),
        df["movie_id"].nunique(),
    )
    return df


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the u.item movie metadata file.

    Args:
        data_dir: Directory containing the MovieLens raw files.

    Returns:
        DataFrame with columns [movie_id, title, release_year].

    Raises:
        FileNotFoundError: If the movie metadata file is missing.
    """
    items_path = data_dir / "u.item"
    if not items_path.exists():
        raise FileNotFoundError(f"Movie metadata file not found at '{items_path}'.")

    # u.item is pipe-separated; only the first three columns are needed here
    df = pd.read_csv(
        items_path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1, 2],
        names=["movie_id", "title", "release_date"],
        dtype={"movie_id": np.int32},
    )

    # Extract four-digit release year from the 'dd-Mon-YYYY' date string
    df["release_year"] = (
        df["release_date"]
        .str.extract(r"(\d{4})")
        .astype("Int64")  # nullable int to handle NaNs gracefully
    )
    df.drop(columns=["release_date"], inplace=True)
    return df


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Re-index user and movie IDs to zero-based consecutive integers.

    Neural embedding layers require indices in [0, num_entities). The original
    MovieLens IDs are already consecutive but this function guarantees it for
    any derived / filtered subset.

    Args:
        df: Ratings DataFrame with original user_id and movie_id columns.

    Returns:
        Tuple of:
        - df: Updated DataFrame with user_idx and movie_idx columns added.
        - user_map: Mapping {original_user_id → new_user_idx}.
        - movie_map: Mapping {original_movie_id → new_movie_idx}.
    """
    unique_users = sorted(df["user_id"].unique())
    unique_movies = sorted(df["movie_id"].unique())

    user_map: dict[int, int] = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_map: dict[int, int] = {mid: idx for idx, mid in enumerate(unique_movies)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map).astype(np.int64)
    df["movie_idx"] = df["movie_id"].map(movie_map).astype(np.int64)

    return df, user_map, movie_map


def normalize_ratings(
    df: pd.DataFrame,
    min_r: float = 1.0,
    max_r: float = 5.0,
) -> pd.DataFrame:
    """Scale ratings to [0, 1] using min-max normalization.

    Args:
        df:    DataFrame that contains a 'rating' column.
        min_r: Minimum possible rating value.
        max_r: Maximum possible rating value.

    Returns:
        DataFrame with an additional 'rating_norm' column.
    """
    df = df.copy()
    df["rating_norm"] = (df["rating"] - min_r) / (max_r - min_r)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into train / validation / test sets.

    The split is stratified by user so every user has ratings in each split.

    Args:
        df:           Full ratings DataFrame.
        test_size:    Fraction of data reserved for testing.
        val_size:     Fraction of *remaining* data reserved for validation.
        random_state: RNG seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    # val_size is expressed relative to the train+val pool
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, random_state=random_state, shuffle=True
    )

    logger.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────


class RatingsDataset(Dataset):
    """Thin wrapper around a ratings DataFrame for use with DataLoader.

    Each sample is a triple (user_idx, movie_idx, normalized_rating).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the dataset from a DataFrame.

        Args:
            df: DataFrame with columns [user_idx, movie_idx, rating_norm].
        """
        required_cols = {"user_idx", "movie_idx", "rating_norm"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # Store as contiguous tensors for fast indexing
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating_norm"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.movies[idx], self.ratings[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model: Neural Matrix Factorization (NMF)
# ─────────────────────────────────────────────────────────────────────────────


class NeuralMatrixFactorization(nn.Module):
    """Neural Matrix Factorization model combining GMF and MLP branches.

    Architecture overview:
        GMF branch  → element-wise product of user & item embeddings
        MLP branch  → concatenation of user & item embeddings, passed through
                       a stack of fully-connected layers with BatchNorm + Dropout
        Output      → linear combination of GMF and MLP outputs → sigmoid → [0,1]

    Args:
        num_users:        Total number of unique users.
        num_movies:       Total number of unique movies.
        gmf_dim:          Embedding dimension for the GMF branch.
        mlp_dim:          Embedding dimension per side (user / movie) for MLP.
        mlp_hidden_dims:  Hidden-layer sizes for the MLP tower.
        dropout:          Dropout probability applied inside MLP layers.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64, 32]

        # ── GMF embeddings ────────────────────────────────────────────────────
        self.gmf_user_emb = nn.Embedding(num_users, gmf_dim)
        self.gmf_movie_emb = nn.Embedding(num_movies, gmf_dim)

        # ── MLP embeddings ────────────────────────────────────────────────────
        self.mlp_user_emb = nn.Embedding(num_users, mlp_dim)
        self.mlp_movie_emb = nn.Embedding(num_movies, mlp_dim)

        # ── MLP tower ─────────────────────────────────────────────────────────
        mlp_layers: list[nn.Module] = []
        in_size = mlp_dim * 2  # user emb + movie emb concatenated
        for out_size in mlp_hidden_dims:
            mlp_layers += [
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = out_size
        self.mlp_tower = nn.Sequential(*mlp_layers)

        # ── Final prediction layer ─────────────────────────────────────────────
        # Input = GMF output (gmf_dim) + MLP output (last hidden dim)
        self.output_layer = nn.Linear(gmf_dim + mlp_hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Initialise weights using Xavier uniform for stability
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to linear layers and normal init to embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        """Compute predicted (normalised) ratings for a batch.

        Args:
            user_idx:  Long tensor of user indices, shape (B,).
            movie_idx: Long tensor of movie indices, shape (B,).

        Returns:
            Float tensor of predicted normalised ratings, shape (B,).
        """
        # ── GMF branch ────────────────────────────────────────────────────────
        gmf_u = self.gmf_user_emb(user_idx)  # (B, gmf_dim)
        gmf_m = self.gmf_movie_emb(movie_idx)  # (B, gmf_dim)
        gmf_out = gmf_u * gmf_m  # element-wise product

        # ── MLP branch ────────────────────────────────────────────────────────
        mlp_u = self.mlp_user_emb(user_idx)  # (B, mlp_dim)
        mlp_m = self.mlp_movie_emb(movie_idx)  # (B, mlp_dim)
        mlp_in = torch.cat([mlp_u, mlp_m], dim=-1)  # (B, mlp_dim*2)
        mlp_out = self.mlp_tower(mlp_in)  # (B, last_hidden_dim)

        # ── Combine & predict ─────────────────────────────────────────────────
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        pred = self.sigmoid(self.output_layer(combined))
        return pred.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        preds:   Predicted rating values (original scale).
        targets: Ground-truth rating values.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        preds:   Predicted rating values.
        targets: Ground-truth rating values.

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(preds - targets)))


def precision_recall_ndcg_at_k(
    model: nn.Module,
    df: pd.DataFrame,
    num_movies: int,
    k: int = 10,
    threshold: float = 3.5,
    max_users: int = 200,
) -> dict[str, float]:
    """Compute Precision@K, Recall@K, and NDCG@K averaged over users.

    For each user the model scores all movies not seen in *df* (leave-one-out
    is not used here; instead the test set itself acts as the ground-truth
    relevant set).

    Args:
        model:      Trained NMF model in eval mode.
        df:         Test DataFrame with columns [user_idx, movie_idx, rating].
        num_movies: Total number of unique movies in the catalogue.
        k:          Cut-off rank for the metrics.
        threshold:  Minimum *original-scale* rating to consider a movie relevant.
        max_users:  Cap on the number of users evaluated (for speed).

    Returns:
        Dict with keys 'precision@k', 'recall@k', 'ndcg@k'.
    """
    model.eval()
    precisions, recalls, ndcgs = [], [], []

    # Build a set of (user, movie) pairs the user has already rated
    rated_pairs: set[tuple[int, int]] = set(zip(df["user_idx"], df["movie_idx"]))

    # Restrict evaluation to users present in df for speed
    sample_users = df["user_idx"].unique()[:max_users]

    with torch.no_grad():
        for uid in sample_users:
            # Ground-truth: movies rated ≥ threshold by this user
            user_rows = df[df["user_idx"] == uid]
            relevant = set(
                user_rows.loc[user_rows["rating"] >= threshold, "movie_idx"].values
            )
            if not relevant:
                # Skip users with no relevant items in the test set
                continue

            # Score all catalogue movies that the user has NOT rated in *any* split
            all_movies = torch.arange(num_movies, device=DEVICE)
            user_tensor = torch.full(
                (num_movies,), uid, dtype=torch.long, device=DEVICE
            )
            scores = model(user_tensor, all_movies).cpu().numpy()

            # Rank movies by descending predicted score
            top_k_indices = np.argsort(-scores)[:k]

            hits = [1 if idx in relevant else 0 for idx in top_k_indices]

            # Precision@K
            precisions.append(sum(hits) / k)

            # Recall@K
            recalls.append(sum(hits) / max(len(relevant), 1))

            # NDCG@K – discount via log2(rank+2) to avoid log(1)=0 at rank 0
            dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one full training epoch.

    Args:
        model:     NMF model.
        loader:    DataLoader for the training split.
        optimizer: Optimiser instance.
        criterion: Loss function (e.g., MSELoss).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for users, movies, ratings in loader:
        users = users.to(DEVICE)
        movies = movies.to(DEVICE)
        ratings = ratings.to(DEVICE)

        optimizer.zero_grad()
        preds = model(users, movies)
        loss = criterion(preds, ratings)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(ratings)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Evaluate model on a data loader.

    Args:
        model:     NMF model in eval mode.
        loader:    DataLoader for validation or test split.
        criterion: Loss function.

    Returns:
        Tuple of (avg_loss, rmse_score, mae_score) in *original* rating scale.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for users, movies, ratings in loader:
        users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
        preds = model(users, movies)
        loss = criterion(preds, ratings)
        total_loss += loss.item() * len(ratings)

        # Denormalize from [0,1] back to the original [1,5] scale
        all_preds.append((preds.cpu().numpy() * 4.0) + 1.0)
        all_targets.append((ratings.cpu().numpy() * 4.0) + 1.0)

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    return (
        total_loss / len(loader.dataset),
        rmse(preds_arr, targets_arr),
        mae(preds_arr, targets_arr),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
) -> dict[str, list[float]]:
    """Full training loop with early stopping and LR scheduling.

    Args:
        model:        NMF model.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        num_epochs:   Maximum number of epochs to train.
        lr:           Initial learning rate for Adam.
        weight_decay: L2 regularisation coefficient.
        patience:     Early-stopping patience (epochs without val improvement).

    Returns:
        Dict containing per-epoch lists: 'train_loss', 'val_loss',
        'val_rmse', 'val_mae'.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Reduce LR by ×0.5 if val_loss doesn't improve for `patience` epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
    }

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve_count = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_rmse_score, val_mae_score = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse_score)
        history["val_mae"].append(val_mae_score)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %02d/%02d | train_loss=%.5f | val_loss=%.5f | "
            "RMSE=%.4f | MAE=%.4f | lr=%.2e | %.1fs",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            val_rmse_score,
            val_mae_score,
            lr_now,
            elapsed,
        )

        # Save the best model weights observed so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    logger.info("Restored best model with val_loss=%.5f", best_val_loss)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to the PLOTS_DIR directory.

    Args:
        fig:  Matplotlib Figure object to save.
        name: Filename (without directory prefix) for the PNG output.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("Saved plot → %s", path)
    plt.close(fig)


def plot_rating_distribution(df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    """Visualise the rating distribution and top-10 most-rated movies.

    Args:
        df:        Ratings DataFrame.
        movies_df: Movie metadata DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview", fontsize=15, fontweight="bold", color="#e0e0f5")

    # ── Left: Rating frequency bar chart ──────────────────────────────────────
    ax = axes[0]
    counts = df["rating"].value_counts().sort_index()
    bars = ax.bar(
        counts.index, counts.values, color=ACCENT_COLORS[0], width=0.6, zorder=3
    )
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Ratings")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", zorder=2)

    # Annotate each bar with its count
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#c8c8e0",
        )

    # ── Right: Top-10 most-rated movies (horizontal bar) ──────────────────────
    ax = axes[1]
    top_movies = (
        df.groupby("movie_id")["rating"]
        .count()
        .nlargest(10)
        .reset_index()
        .merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
    )
    # Truncate long titles for readability
    top_movies["label"] = top_movies["title"].str[:30]
    top_movies = top_movies.sort_values("rating")

    bars = ax.barh(
        top_movies["label"],
        top_movies["rating"],
        color=ACCENT_COLORS[1],
        height=0.6,
        zorder=3,
    )
    ax.set_title("Top 10 Most-Rated Movies")
    ax.set_xlabel("Number of Ratings")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="x", zorder=2)

    for bar, val in zip(bars, top_movies["rating"]):
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            str(int(val)),
            va="center",
            fontsize=8,
            color="#c8c8e0",
        )

    plt.tight_layout()
    _save(fig, "01_rating_distribution.png")


def plot_user_activity(df: pd.DataFrame) -> None:
    """Plot the distribution of ratings per user and per movie.

    Args:
        df: Ratings DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Activity Distribution (Long-Tail Effect)",
        fontsize=15,
        fontweight="bold",
        color="#e0e0f5",
    )

    for ax, col, label, color in zip(
        axes,
        ["user_id", "movie_id"],
        ["Ratings per User", "Ratings per Movie"],
        [ACCENT_COLORS[2], ACCENT_COLORS[3]],
    ):
        counts = df.groupby(col).size()
        ax.hist(counts, bins=50, color=color, alpha=0.85, edgecolor="none", zorder=3)
        ax.set_title(label)
        ax.set_xlabel("Number of Ratings")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")  # log-scale reveals the long tail clearly
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="y", zorder=2)
        ax.axvline(
            counts.median(),
            color="#f7b731",
            linestyle="--",
            linewidth=1.2,
            label=f"Median = {counts.median():.0f}",
        )
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "02_activity_distribution.png")


def plot_training_history(history: dict[str, list[float]]) -> None:
    """Plot training / validation loss curves and RMSE/MAE over epochs.

    Args:
        history: Dict returned by the train() function.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold", color="#e0e0f5")

    # ── Loss curves ───────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(
        epochs,
        history["train_loss"],
        color=ACCENT_COLORS[0],
        linewidth=2,
        label="Train Loss",
    )
    ax.plot(
        epochs,
        history["val_loss"],
        color=ACCENT_COLORS[1],
        linewidth=2,
        linestyle="--",
        label="Val Loss",
    )
    ax.set_title("MSE Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # ── RMSE / MAE curves ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(
        epochs,
        history["val_rmse"],
        color=ACCENT_COLORS[2],
        linewidth=2,
        label="Val RMSE",
    )
    ax.plot(
        epochs,
        history["val_mae"],
        color=ACCENT_COLORS[3],
        linewidth=2,
        linestyle="--",
        label="Val MAE",
    )
    best_epoch = int(np.argmin(history["val_rmse"])) + 1
    ax.axvline(
        best_epoch,
        color="#ffffff",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label=f"Best epoch ({best_epoch})",
    )
    ax.set_title("RMSE & MAE on Validation Set")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (original scale)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    _save(fig, "03_training_history.png")


def plot_prediction_analysis(
    model: nn.Module,
    test_loader: DataLoader,
) -> None:
    """Scatter plot of predicted vs actual ratings + residuals histogram.

    Args:
        model:       Trained NMF model.
        test_loader: DataLoader for the test split.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for users, movies, ratings in test_loader:
            preds = model(users.to(DEVICE), movies.to(DEVICE)).cpu().numpy()
            all_preds.extend((preds * 4.0) + 1.0)
            all_targets.extend((ratings.numpy() * 4.0) + 1.0)

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    residuals = preds_arr - targets_arr

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Prediction Analysis on Test Set",
        fontsize=15,
        fontweight="bold",
        color="#e0e0f5",
    )

    # ── Predicted vs Actual scatter ───────────────────────────────────────────
    ax = axes[0]
    ax.scatter(targets_arr, preds_arr, alpha=0.08, s=6, color=ACCENT_COLORS[0])
    ax.plot(
        [1, 5],
        [1, 5],
        color=ACCENT_COLORS[1],
        linewidth=1.5,
        linestyle="--",
        label="Perfect prediction",
    )
    ax.set_title("Predicted vs Actual Rating")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.legend()
    ax.grid(True)

    # ── Residuals histogram ───────────────────────────────────────────────────
    ax = axes[1]
    ax.hist(
        residuals,
        bins=60,
        color=ACCENT_COLORS[2],
        edgecolor="none",
        alpha=0.9,
        zorder=3,
    )
    ax.axvline(0, color=ACCENT_COLORS[1], linewidth=1.5, linestyle="--")
    ax.axvline(
        residuals.mean(),
        color=ACCENT_COLORS[3],
        linewidth=1.2,
        linestyle="-",
        label=f"Mean residual: {residuals.mean():.3f}",
    )
    ax.set_title("Residuals Distribution")
    ax.set_xlabel("Prediction Error (Predicted − Actual)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", zorder=2)

    plt.tight_layout()
    _save(fig, "04_prediction_analysis.png")


def plot_embedding_similarity(
    model: nn.Module,
    movies_df: pd.DataFrame,
    movie_map: dict[int, int],
    top_n: int = 30,
) -> None:
    """Heatmap of cosine similarity among the top-N most-rated movie embeddings.

    Args:
        model:     Trained NMF model.
        movies_df: Movie metadata DataFrame.
        movie_map: Mapping {original_movie_id → new_movie_idx}.
        top_n:     Number of movies to include in the heatmap.
    """
    model.eval()

    # Use GMF embeddings (pure MF signal, most interpretable)
    emb_matrix = (
        model.gmf_movie_emb.weight.detach().cpu().numpy()
    )  # (num_movies, gmf_dim)

    # Select the top_n most-rated movies based on movie_id ordering
    top_ids = sorted(movie_map.keys())[:top_n]
    top_idxs = [movie_map[mid] for mid in top_ids]
    selected_embs = emb_matrix[top_idxs]  # (top_n, gmf_dim)

    # Cosine similarity matrix
    norms = np.linalg.norm(selected_embs, axis=1, keepdims=True) + 1e-9
    normed = selected_embs / norms
    sim_matrix = normed @ normed.T  # (top_n, top_n)

    # Fetch movie labels (truncated titles)
    id_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
    labels = [id_to_title.get(mid, str(mid))[:20] for mid in top_ids]

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle(
        f"Movie Embedding Cosine Similarity (Top {top_n})",
        fontsize=14,
        fontweight="bold",
        color="#e0e0f5",
        y=1.01,
    )

    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="magma",
        vmin=-0.5,
        vmax=1.0,
        ax=ax,
        square=True,
        linewidths=0.3,
        linecolor="#0f0f1a",
        cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"},
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    _save(fig, "05_embedding_similarity.png")


def plot_metrics_summary(metrics: dict[str, float]) -> None:
    """Horizontal bar chart of final evaluation metrics.

    Args:
        metrics: Dict of metric_name → value to visualise.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Final Model Metrics", fontsize=14, fontweight="bold", color="#e0e0f5")

    names = list(metrics.keys())
    values = list(metrics.values())
    colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(names))]

    bars = ax.barh(names, values, color=colors, height=0.5, zorder=3)
    ax.set_xlabel("Value")
    ax.grid(axis="x", zorder=2)
    ax.set_xlim(0, max(values) * 1.25)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
            color="#e0e0f5",
        )

    plt.tight_layout()
    _save(fig, "06_metrics_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────


def recommend_for_user(
    model: nn.Module,
    user_id: int,
    user_map: dict[int, int],
    movie_map: dict[int, int],
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Generate top-K movie recommendations for a given user.

    Movies the user has already rated are excluded from the recommendation list.

    Args:
        model:      Trained NMF model in eval mode.
        user_id:    Original MovieLens user ID (1-based).
        user_map:   Mapping {original_user_id → user_idx}.
        movie_map:  Mapping {original_movie_id → movie_idx}.
        ratings_df: Full ratings DataFrame (used to filter seen movies).
        movies_df:  Movie metadata DataFrame for title lookup.
        top_k:      Number of recommendations to return.

    Returns:
        DataFrame with columns [movie_id, title, release_year, predicted_rating],
        sorted by predicted_rating descending.

    Raises:
        KeyError: If user_id is not present in the training data.
        ValueError: If top_k is not a positive integer.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")
    if user_id not in user_map:
        raise KeyError(
            f"User ID {user_id} was not seen during training. "
            f"Valid range: {min(user_map)}-{max(user_map)}."
        )

    model.eval()
    user_idx = user_map[user_id]

    # Determine movies the user has already interacted with (all splits)
    seen_movies: set[int] = set(
        ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].values
    )

    # Candidate movies = catalogue minus seen
    candidate_movie_ids = [mid for mid in movie_map if mid not in seen_movies]
    candidate_idxs = [movie_map[mid] for mid in candidate_movie_ids]

    user_tensor = torch.tensor(
        [user_idx] * len(candidate_idxs), dtype=torch.long, device=DEVICE
    )
    movie_tensor = torch.tensor(candidate_idxs, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        scores = model(user_tensor, movie_tensor).cpu().numpy()

    # Denormalize to original [1,5] scale
    predicted_ratings = scores * 4.0 + 1.0

    # Build result DataFrame
    results = pd.DataFrame(
        {
            "movie_id": candidate_movie_ids,
            "predicted_rating": predicted_ratings,
        }
    )
    results = (
        results.nlargest(top_k, "predicted_rating")
        .merge(
            movies_df[["movie_id", "title", "release_year"]], on="movie_id", how="left"
        )[["movie_id", "title", "release_year", "predicted_rating"]]
        .reset_index(drop=True)
    )
    results.index += 1  # 1-based rank for display
    return results


def find_similar_movies(
    model: nn.Module,
    movie_id: int,
    movie_map: dict[int, int],
    movies_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Find the top-K most similar movies using GMF embedding cosine similarity.

    Args:
        model:     Trained NMF model.
        movie_id:  Original MovieLens movie ID for the query movie.
        movie_map: Mapping {original_movie_id → movie_idx}.
        movies_df: Movie metadata DataFrame.
        top_k:     Number of similar movies to return.

    Returns:
        DataFrame with columns [movie_id, title, release_year, similarity],
        sorted by similarity descending (query movie excluded).

    Raises:
        KeyError: If movie_id is not present in the catalogue.
    """
    if movie_id not in movie_map:
        raise KeyError(
            f"Movie ID {movie_id} not found. "
            f"Valid range: {min(movie_map)}-{max(movie_map)}."
        )

    emb_matrix = model.gmf_movie_emb.weight.detach().cpu().numpy()
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
    normed = emb_matrix / norms  # (num_movies, gmf_dim)

    query_idx = movie_map[movie_id]
    query_vec = normed[query_idx]  # (gmf_dim,)
    sims = normed @ query_vec  # (num_movies,)

    # Exclude the query movie itself
    sims[query_idx] = -1.0

    top_idxs = np.argsort(-sims)[:top_k]
    # Reverse mapping: movie_idx → original movie_id
    idx_to_mid = {v: k for k, v in movie_map.items()}

    similar_ids = [idx_to_mid[i] for i in top_idxs]
    similar_sims = [float(sims[i]) for i in top_idxs]

    results = pd.DataFrame({"movie_id": similar_ids, "similarity": similar_sims})
    results = results.merge(
        movies_df[["movie_id", "title", "release_year"]], on="movie_id", how="left"
    )[["movie_id", "title", "release_year", "similarity"]].reset_index(drop=True)
    results.index += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Execute the complete recommendation pipeline end-to-end.

    Steps:
        1. Download & load the MovieLens 100K dataset.
        2. Explore and visualise the raw data.
        3. Preprocess: encode IDs, normalise ratings, split into splits.
        4. Build DataLoaders and instantiate the NMF model.
        5. Train with early stopping and learning-rate scheduling.
        6. Evaluate on the held-out test set (RMSE, MAE, Precision/Recall/NDCG).
        7. Generate inference examples (user recommendations, similar movies).
        8. Save all visualisations to disk.
    """
    logger.info("=" * 60)
    logger.info("Movie Recommendation System — Neural Matrix Factorization")
    logger.info("Device: %s", DEVICE)
    logger.info("=" * 60)

    # ── 1. Data acquisition ───────────────────────────────────────────────────
    download_movielens()
    ratings_df = load_ratings()
    movies_df = load_movies()

    # ── 2. EDA visualisations ─────────────────────────────────────────────────
    logger.info("Generating exploratory data analysis plots …")
    plot_rating_distribution(ratings_df, movies_df)
    plot_user_activity(ratings_df)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    ratings_df, user_map, movie_map = encode_ids(ratings_df)
    ratings_df = normalize_ratings(ratings_df)
    train_df, val_df, test_df = split_data(ratings_df)

    num_users = len(user_map)
    num_movies = len(movie_map)
    logger.info("Catalogue size: %d users | %d movies", num_users, num_movies)

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    batch_size = 1024
    train_loader = DataLoader(
        RatingsDataset(train_df), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        RatingsDataset(val_df), batch_size=batch_size * 2, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        RatingsDataset(test_df), batch_size=batch_size * 2, shuffle=False, num_workers=0
    )

    # ── 5. Model instantiation ────────────────────────────────────────────────
    model = NeuralMatrixFactorization(
        num_users=num_users,
        num_movies=num_movies,
        gmf_dim=32,
        mlp_dim=32,
        mlp_hidden_dims=[128, 64, 32],
        dropout=0.2,
    ).to(DEVICE)

    # Log parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── 6. Training ───────────────────────────────────────────────────────────
    history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=30,
        lr=1e-3,
        weight_decay=1e-5,
        patience=6,
    )
    plot_training_history(history)

    # ── 7. Test-set evaluation ────────────────────────────────────────────────
    criterion = nn.MSELoss()
    _, test_rmse, test_mae = evaluate(model, test_loader, criterion)
    ranking_metrics = precision_recall_ndcg_at_k(
        model, test_df, num_movies, k=10, threshold=3.5, max_users=200
    )

    all_metrics = {
        "RMSE": test_rmse,
        "MAE": test_mae,
        **ranking_metrics,
    }

    logger.info("=" * 40)
    logger.info("TEST SET METRICS")
    logger.info("=" * 40)
    for metric, value in all_metrics.items():
        logger.info("  %-20s %.4f", metric, value)
    logger.info("=" * 40)

    # ── 8. Post-training visualisations ───────────────────────────────────────
    plot_prediction_analysis(model, test_loader)
    plot_embedding_similarity(model, movies_df, movie_map, top_n=30)
    plot_metrics_summary(all_metrics)

    # ── 9. Inference examples ─────────────────────────────────────────────────
    logger.info("\n%s", "─" * 60)
    logger.info("INFERENCE — Top-10 recommendations for user 1")
    logger.info("─" * 60)
    recs = recommend_for_user(
        model,
        user_id=1,
        user_map=user_map,
        movie_map=movie_map,
        ratings_df=ratings_df,
        movies_df=movies_df,
        top_k=10,
    )
    print(recs.to_string())

    logger.info("\n%s", "─" * 60)
    logger.info("INFERENCE — Top-10 similar movies to movie ID 1 (Toy Story)")
    logger.info("─" * 60)
    similar = find_similar_movies(
        model,
        movie_id=1,
        movie_map=movie_map,
        movies_df=movies_df,
        top_k=10,
    )
    print(similar.to_string())

    logger.info("\nAll plots saved to '%s/'", PLOTS_DIR)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
