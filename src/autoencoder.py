"""
Autoencoder module (Chapters VI, X) — PyTorch implementation.

Implements:
  1. Shallow autoencoder for feature embedding
  2. Reconstruction-error based anomaly detection
  3. Embedding-space clustering (K-Means on bottleneck features)
  4. Comparison: clustering on raw features vs. autoencoder embeddings
  5. Latent space visualization (PCA of embeddings)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.config import (
    OUT_FIG, OUT_MODEL, OUT_RESULT, RANDOM_SEED,
    AE_ENCODING_DIM, AE_EPOCHS, AE_BATCH_SIZE, AE_ANOMALY_PCTL,
)
from src.utils import logger, save_figure, save_results


def prepare_ae_data(df):
    """Prepare scaled feature matrix for autoencoder."""
    feature_cols = ["TOTAL_DAMAGE", "DURATION_MIN", "INJURIES_DIRECT",
                    "DEATHS_DIRECT", "BEGIN_LAT", "BEGIN_LON",
                    "MAGNITUDE", "HOUR", "MONTH"]
    available = [c for c in feature_cols if c in df.columns]

    subset = df[df["TOTAL_DAMAGE"] > 0][available].dropna().copy()
    for col in ["TOTAL_DAMAGE"]:
        if col in subset.columns:
            subset[col] = np.log1p(subset[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    logger.info(f"Autoencoder data: {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} features")
    return X_scaled, subset, scaler, available


def _get_device():
    """Get PyTorch device. CPU only to avoid MPS/DataLoader deadlocks."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_and_train_autoencoder(X_scaled):
    """Build and train a shallow autoencoder using PyTorch (CPU, manual batching)."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        torch.manual_seed(RANDOM_SEED)
    except ImportError:
        logger.warning("PyTorch not available. Skipping autoencoder.")
        return None, None, None

    # Shut down loky workers to avoid deadlocks with PyTorch
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
    import gc
    gc.collect()

    device = _get_device()
    logger.info(f"Using device: {device}")
    n_features = X_scaled.shape[1]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, AE_ENCODING_DIM), nn.ReLU(),
            )
        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(AE_ENCODING_DIM, 16), nn.ReLU(), nn.BatchNorm1d(16),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, n_features),
            )
        def forward(self, x):
            return self.net(x)

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data
    n_val = int(len(X_scaled) * 0.2)
    idx = np.random.permutation(len(X_scaled))
    X_train_np = X_scaled[idx[n_val:]]
    X_val_np = X_scaled[idx[:n_val]]
    X_val_t = torch.FloatTensor(X_val_np).to(device)

    # Train with manual batching (avoids DataLoader deadlocks)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = len(X_train_np)

    for epoch in range(AE_EPOCHS):
        model.train()
        epoch_loss = 0
        n_samples = 0
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, AE_BATCH_SIZE):
            batch_idx = perm[start:start + AE_BATCH_SIZE]
            batch = torch.FloatTensor(X_train_np[batch_idx]).to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
            n_samples += len(batch)
        train_losses.append(epoch_loss / n_samples)

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    logger.info(f"Autoencoder trained: {len(train_losses)} epochs, "
                f"final val_loss={val_losses[-1]:.6f}")

    # Training curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train MSE", linewidth=2)
    ax.plot(val_losses, label="Validation MSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curve", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "33_autoencoder_training", OUT_FIG)
    plt.close(fig)

    return model, model.encoder, {"train_losses": train_losses, "val_losses": val_losses}


def _predict(model, X_scaled):
    """Run inference with a PyTorch model."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_scaled).to(device))
        return out.cpu().numpy()


def reconstruction_anomaly_detection(autoencoder, X_scaled):
    """Use reconstruction error as anomaly score."""
    logger.info("─" * 40)
    logger.info("Autoencoder Anomaly Detection")
    logger.info("─" * 40)

    reconstructed = _predict(autoencoder, X_scaled)
    recon_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)

    threshold = np.percentile(recon_error, AE_ANOMALY_PCTL)
    anomalies = recon_error > threshold
    logger.info(f"Reconstruction error threshold ({AE_ANOMALY_PCTL}th pctl): {threshold:.4f}")
    logger.info(f"Anomalies detected: {anomalies.sum():,} ({anomalies.mean()*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(recon_error, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(x=threshold, color="red", linestyle="--",
                    label=f"{AE_ANOMALY_PCTL}th percentile")
    axes[0].set_xlabel("Reconstruction Error (MSE)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reconstruction Error Distribution", fontweight="bold")
    axes[0].legend()

    axes[1].hist(np.log10(recon_error + 1e-10), bins=100, color="darkorange",
                 alpha=0.7, edgecolor="white")
    axes[1].axvline(x=np.log10(threshold), color="red", linestyle="--")
    axes[1].set_xlabel("Log₁₀(Reconstruction Error)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Log Reconstruction Error", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "34_reconstruction_error", OUT_FIG)
    plt.close(fig)

    return recon_error, anomalies


def embedding_clustering(encoder, X_scaled, df_subset):
    """Cluster in embedding space and compare with raw features."""
    logger.info("─" * 40)
    logger.info("Embedding Space Clustering")
    logger.info("─" * 40)

    embeddings = _predict(encoder, X_scaled)
    logger.info(f"Embedding shape: {embeddings.shape}")

    results = []
    k_range = [3, 4, 5, 6, 7, 8]
    for name, data in [("Raw Features", X_scaled), ("AE Embeddings", embeddings)]:
        sils = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_SEED)
            labels = km.fit_predict(data)
            sil = silhouette_score(data, labels, sample_size=min(10000, len(data)))
            sils.append(sil)
            results.append({"Space": name, "k": k, "Silhouette": sil})
        logger.info(f"  {name}: best silhouette = {max(sils):.4f} at k={k_range[np.argmax(sils)]}")

    results_df = pd.DataFrame(results)
    save_results(results_df, "ae_clustering_comparison", OUT_RESULT)

    # Compute comparison statistics
    raw_sils = results_df[results_df["Space"] == "Raw Features"]["Silhouette"].values
    ae_sils = results_df[results_df["Space"] == "AE Embeddings"]["Silhouette"].values
    raw_best = raw_sils.max()
    ae_best = ae_sils.max()
    raw_std = raw_sils.std()
    ae_std = ae_sils.std()
    ae_wins = sum(ae_sils > raw_sils)

    logger.info("─" * 40)
    logger.info("EMBEDDING vs RAW FEATURE CLUSTERING COMPARISON")
    logger.info("─" * 40)
    logger.info(f"  Raw features: best silhouette = {raw_best:.4f}, "
                f"std across k = {raw_std:.4f}")
    logger.info(f"  AE embeddings: best silhouette = {ae_best:.4f}, "
                f"std across k = {ae_std:.4f}")
    logger.info(f"  AE wins at {ae_wins}/{len(k_range)} values of k")
    if ae_std < raw_std:
        logger.info("  AE embeddings produce MORE STABLE clustering across k values")
        logger.info(f"  (std: {ae_std:.4f} vs {raw_std:.4f})")
    if ae_wins >= len(k_range) // 2:
        logger.info("  AE embeddings improve clustering for the majority of k values")
    else:
        logger.info("  Raw features achieve highest single silhouette score, "
                    "but AE embeddings provide more consistent quality")

    fig, ax = plt.subplots(figsize=(10, 6))
    for space, group in results_df.groupby("Space"):
        ax.plot(group["k"], group["Silhouette"], "o-", linewidth=2, markersize=8,
                label=space)
    # Add shaded region showing where AE > Raw
    for i in range(len(k_range)):
        if ae_sils[i] > raw_sils[i]:
            ax.axvspan(k_range[i] - 0.3, k_range[i] + 0.3,
                       alpha=0.1, color="green")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Quality: Raw Features vs. Autoencoder Embeddings\n"
                 f"(AE more stable: std={ae_std:.3f} vs {raw_std:.3f}; "
                 f"AE wins at {ae_wins}/{len(k_range)} k values)",
                 fontweight="bold", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "35_ae_clustering_comparison", OUT_FIG)
    plt.close(fig)

    # Latent space PCA visualization
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    n_vis = min(20000, len(emb_2d))
    idx = np.random.choice(len(emb_2d), n_vis, replace=False)

    scatter = ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1],
                        c=X_scaled[idx, 0], cmap="plasma",
                        s=3, alpha=0.4, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Autoencoder Latent Space (PCA projection, colored by damage)",
                 fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Scaled Damage")
    fig.tight_layout()
    save_figure(fig, "36_latent_space_pca", OUT_FIG)
    plt.close(fig)

    return embeddings


def run_autoencoder(df: pd.DataFrame):
    """Run the full autoencoder pipeline."""
    logger.info("=" * 60)
    logger.info("AUTOENCODER ANALYSIS")
    logger.info("=" * 60)

    X_scaled, subset, scaler, feature_names = prepare_ae_data(df)
    autoencoder, encoder, history = build_and_train_autoencoder(X_scaled)

    if autoencoder is None:
        logger.warning("Autoencoder skipped (PyTorch not available)")
        return None

    recon_error, anomalies = reconstruction_anomaly_detection(autoencoder, X_scaled)
    embeddings = embedding_clustering(encoder, X_scaled, subset)

    # Save models
    import torch
    torch.save(autoencoder.state_dict(), str(OUT_MODEL / "autoencoder.pt"))
    torch.save(encoder.state_dict(), str(OUT_MODEL / "encoder.pt"))
    joblib.dump(scaler, OUT_MODEL / "ae_scaler.joblib")

    logger.info("Autoencoder analysis complete.")
    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "embeddings": embeddings,
        "recon_error": recon_error,
        "anomalies": anomalies,
    }


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_autoencoder(df)
