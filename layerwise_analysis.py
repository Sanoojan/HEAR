import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score



datasets = [
    "DFDC",
    "DFDCP",
    "Celeb-DF-v2",
    "DeepFakeDetection",
    "WDF"
]

model_path="/egr/research-sprintai/baliahsa/projects/DeepfakeBench/visualizations/saved_patch_scores/Clip_L_inference5_PatchClS_TripleStream_3DConv_TopKFrame_1_0_0_FM_0.6_CE_0.0_MIL_1_Var_0_DC_Layer_analysis/{}.pkl"


print("\n===== LAYER-WISE ANALYSIS =====")

layerwise_aucs={d: [] for d in datasets}

for dataset in datasets:
    
    path = model_path.format(dataset)

    if not os.path.exists(path):
        continue

    model_outs = pickle.load(open(path, 'rb'))

    labels = np.asarray(model_outs['labels'])
    layer_scores = model_outs['layerwise_prediction']  # Assuming this is a dict of {layer_name: scores}
    layer_scores = np.asarray(layer_scores)  # Convert to numpy array for easier processing
    layer_scores= layer_scores.reshape(labels.shape[0], -1,2)  # Flatten if necessary (e.g., if scores are per-patch, average them)
    # permute to (num_layers, num_samples,2)
    layer_scores = np.transpose(layer_scores, (1, 0, 2))  # Now shape is (num_samples, num_layers, 2)
    # apply softmax to get probabilities
    layer_scores = np.exp(layer_scores) / np.sum(np.exp(layer_scores), axis=-1, keepdims=True)  # Shape: (num_samples, num_layers, 2)
    layer_scores = layer_scores[:,:,1]  # Take the probability of the positive class, shape: (num_samples, num_layers)
    mean_layer_scores = np.mean(layer_scores, axis=0)  # Average over samples to get mean score per layer, shape: (num_layers,)
    print(f"\nDataset: {dataset}")
    # breakpoint()
    
    
    for layer_name, scores in enumerate(layer_scores):
        auc = roc_auc_score(labels, scores)
        print(f"  Layer: {layer_name} | AUC: {auc:.4f}")
        layerwise_aucs[dataset].append(auc)
    auc_mean = roc_auc_score(labels, mean_layer_scores)
    print(f"  Mean Layer Score AUC: {auc_mean:.4f}")
    layerwise_aucs[dataset].append(auc_mean)
# breakpoint()
# Plotting
plt.figure(figsize=(12, 6))
for dataset in datasets:
    plt.plot(layerwise_aucs[dataset], label=dataset, marker='o')
plt.title("Layer-wise AUC Analysis")
plt.xlabel("Layer Index")
plt.ylabel("AUC")
plt.xticks(range(len(layerwise_aucs[datasets[0]])), [f"Layer {i}" for i in range(len(layerwise_aucs[datasets[0]])-1)] + ["Mean"], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("visualizations/Plots/layerwise_auc_analysis.png")


plt.figure(figsize=(8, 5))

mean_aucs = [
    np.mean([layerwise_aucs[d][i] for d in datasets])
    for i in range(len(layerwise_aucs[datasets[0]]))
]

x = list(range(len(mean_aucs)))

# Plot layer performance (with line)
plt.plot(
    x[:-1],
    mean_aucs[:-1],
    label="Layer Performance",
    marker='o'
)

# Plot ensemble performance (star only, NO line)
plt.plot(
    x[-1],
    mean_aucs[-1],
    marker='*',
    markersize=16,
    linestyle='None',
    label="Cross-Layer score Averaging"
)

plt.title("Mean AUC Across Datasets")
plt.xlabel("Layer Index")
plt.ylabel("Mean AUC")

plt.xticks(
    range(len(mean_aucs)),
    [f"layer {i}" for i in range(len(mean_aucs)-1)] + [""],
    rotation=45
)

plt.grid(True, linestyle='--', alpha=0.5)  # Add grid

plt.legend()
plt.tight_layout()
plt.savefig("visualizations/Plots/layerwise_auc_analysis_mean.png")

print("Ensembled Performance (Final Mean):", mean_aucs[-1])


import matplotlib.pyplot as plt
import numpy as np

# Select datasets to show
selected_datasets = ["Celeb-DF-v2", "DFDC", "DFDCP"]  # Exclude WDF for cleaner plot

# Number of layers (excluding ensemble mean)
num_layers = len(layerwise_aucs[selected_datasets[0]]) - 1

# Extend x-axis by +0.8 for extra spacing
x_layers = np.arange(num_layers)       # layers 0..num_layers-1
x_steam = num_layers + 0.8             # shifted right

plt.figure(figsize=(9,5.5))

# Light colors for individual datasets
light_colors = ["#A6CEE3", "#B2DF8A", "#FB9A99", "#DDA0DD", "#98FB98"]
# Bold color for mean
mean_color = "#1F78B4"

# Plot individual datasets as dashed and lighter
for i, d in enumerate(selected_datasets):
    auc_percent = np.array(layerwise_aucs[d]) * 100  # convert to percent
    plt.plot(
        x_layers,
        auc_percent[:-1],
        marker='o',
        linewidth=1.5,
        linestyle='--',
        color=light_colors[i],
        label=f"{d}"
    )

    # Highlight dataset Avg with star
    plt.scatter(
        x_steam - 0.3,  # slightly left to match dataset star positions
        auc_percent[-1],
        marker='*',
        s=180,
        color=light_colors[i],
        zorder=4
    )

# Compute overall mean across selected datasets
mean_aucs = np.array([
    np.mean([layerwise_aucs[d][i] for d in datasets])
    for i in range(num_layers + 1)]
) * 100  # convert to percent

# Plot overall mean as bold solid line ONLY for layers (exclude last Avg)
plt.plot(
    x_layers,
    mean_aucs[:-1],
    color=mean_color,
    marker='o',
    markersize=8,
    linewidth=2.5,
    label="Mean Across OOD Datasets"
)

# Plot the "HEAR" point separately (no connecting line)
plt.scatter(
    x_steam,
    mean_aucs[-1],
    color=mean_color,
    marker='*',
    s=260,
    label="Cross-Layer Mean (HEAR)",
    zorder=5
)

# X-axis labels with LaTeX + extra spacing for STEAM
xticks = list(x_layers) + [x_steam]
xtick_labels = [f"$\\ell_{{{i+1}}}$" for i in range(num_layers)] + ["HEAR"]
plt.xticks(
    xticks,
    xtick_labels,
    fontsize=11,
    fontweight="bold",
    rotation=0
)
plt.yticks(fontsize=10.5, fontweight="bold")

# Labels and grid
plt.xlabel("Block Index", fontsize=12, fontweight="bold")
plt.ylabel("Video AUC (%)", fontsize=12, fontweight="bold")
plt.ylim(45, 98)
plt.grid(axis="y", linestyle="--", alpha=0.35)

plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("visualizations/Plots/layerwise_auc_selected_datasets_percent.pdf", dpi=300)
plt.show()

print("Overall Ensemble Mean AUC (%):", mean_aucs[-1])