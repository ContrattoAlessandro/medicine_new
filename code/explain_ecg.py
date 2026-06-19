import os
import sys

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from models.resnet1d import resnet1d18
from models.timeseries_utils import butter_filter, apply_butter_filter
from scipy.ndimage import gaussian_filter1d

# Setup plotting style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic support saliency maps for ECG signals.")
    parser.add_argument("--ecg_id", type=int, required=True, help="PTB-XL ECG ID (1-based index).")
    parser.add_argument("--target_class", type=str, default=None, help="The class abbreviation to explain (e.g. NORM, IMI, ASMI). If None, explains the highest predicted class.")
    parser.add_argument("--exp", type=str, default="exp0_filtered_aug_balanced", help="The experiment folder name (e.g. exp0, exp0_filtered_aug_balanced).")
    parser.add_argument("--output", type=str, default=None, help="Output image file path.")
    parser.add_argument("--smooth_sigma", type=float, default=5.0, help="Standard deviation for Gaussian smoothing of gradients.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Saliency threshold (0.0 to 1.0) above which highlights are plotted.")
    parser.add_argument("--color_map", type=str, default="Reds", help="Matplotlib colormap for highlights.")
    args = parser.parse_args()

    # Paths
    data_dir = '../data/ptbxl/'
    output_dir = '../output/'
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading database and scp statements...")
    db = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    db.scp_codes = db.scp_codes.apply(lambda x: ast.literal_eval(x))
    statements = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)

    # Check ecg_id validity
    if args.ecg_id not in db.index:
        print(f"Error: ECG ID {args.ecg_id} not found in database.")
        sys.exit(1)

    row = db.loc[args.ecg_id]
    true_scp_codes = list(row.scp_codes.keys())
    print(f"ECG ID {args.ecg_id} True SCP Codes: {true_scp_codes}")

    # Load raw signal data
    print("Loading raw ECG signals...")
    raw_signals = np.load(os.path.join(data_dir, 'raw100.npy'), allow_pickle=True)
    raw_signal = raw_signals[args.ecg_id - 1]

    # Preprocessing
    fs = 100
    is_filtered_exp = "filtered" in args.exp
    if is_filtered_exp:
        print("Experiment uses filtering. Applying bandpass filter to ECG signal...")
        filter_sos = butter_filter(lowcut=0.5, highcut=40.0, fs=fs, order=5, btype='band')
        signal = apply_butter_filter(raw_signal, filter_sos)
    else:
        print("Experiment does not use filtering. Loading raw ECG signal...")
        signal = raw_signal

    # Standardize signal
    scaler_path = os.path.join(output_dir, args.exp, 'data/standard_scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"Error: Standard scaler not found at {scaler_path}.")
        sys.exit(1)
        
    with open(scaler_path, 'rb') as f:
        ss = pickle.load(f)
    
    # standardize shape (1000, 12)
    standardized_signal = ss.transform(signal.flatten()[:, np.newaxis]).reshape(signal.shape)

    # Transpose to shape (1, 12, 1000) for Conv1d PyTorch model
    X_torch = torch.tensor(standardized_signal.transpose(1, 0), dtype=torch.float32).unsqueeze(0)
    X_torch.requires_grad = True

    # Load multi-label binarizer and model
    mlb_path = os.path.join(output_dir, args.exp, 'data/mlb.pkl')
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    classes = list(mlb.classes_)
    n_classes = len(classes)

    model_checkpoint_path = os.path.join(output_dir, args.exp, 'models/fastai_resnet1d18/models/fastai_resnet1d18.pth')
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}.")
        sys.exit(1)

    print(f"Loading ResNet1d18 model with {n_classes} classes...")
    model = resnet1d18(num_classes=n_classes, input_channels=12, inplanes=128, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128])
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Forward pass
    logits = model(X_torch)
    probs = torch.sigmoid(logits)[0].detach().numpy()

    # Determine target class index
    superclasses_list = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    is_superclass = False
    
    if args.target_class is not None and args.target_class.upper() in superclasses_list:
        is_superclass = True
        superclass_name = args.target_class.upper()
        print(f"Target is identified as a diagnostic superclass: {superclass_name}")
        
        # Retrieve all subclasses belonging to this superclass
        subclasses = statements[statements.diagnostic_class == superclass_name].index.values
        subclass_indices = [classes.index(sc) for sc in subclasses if sc in classes]
        
        if len(subclass_indices) == 0:
            print(f"Error: No subclasses for superclass {superclass_name} found in model classes.")
            sys.exit(1)
            
        # Find which subclass has the highest predicted probability for this record
        target_idx = subclass_indices[np.argmax(probs[subclass_indices])]
        args.target_class = classes[target_idx]
        print(f"Explaining superclass {superclass_name} via its highest-scoring subclass: {args.target_class} (Prob: {probs[target_idx]:.2%})")
    else:
        if args.target_class is not None:
            if args.target_class not in classes:
                print(f"Error: Target class {args.target_class} not found in model classes.")
                sys.exit(1)
            target_idx = classes.index(args.target_class)
        else:
            # Auto-pick class with highest predicted probability
            target_idx = np.argmax(probs)
            args.target_class = classes[target_idx]
            print(f"Auto-selected target class with highest probability: {args.target_class}")

    raw_prob = probs[target_idx]
    cal_prob = raw_prob
    
    # Try loading calibrators
    calibrators_path = os.path.join(output_dir, args.exp, 'data', 'platt_calibrators.pkl')
    if os.path.exists(calibrators_path):
        try:
            with open(calibrators_path, 'rb') as f:
                calibrators = pickle.load(f)
            if target_idx in calibrators:
                # Apply Platt Scaling
                p_clip = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
                raw_logit = np.log(p_clip / (1.0 - p_clip))
                cal_prob = calibrators[target_idx].predict_proba(np.array([[raw_logit]]))[0, 1]
                print(f"Calibration applied: Raw Prob = {raw_prob:.2%} -> Calibrated Prob = {cal_prob:.2%}")
        except Exception as e:
            print(f"Warning: Could not apply calibration: {e}")

    if cal_prob != raw_prob:
        prob_display = f"Calibrated Prob: {cal_prob:.2%} (Raw: {raw_prob:.2%})"
    else:
        prob_display = f"Model Probability: {raw_prob:.2%}"

    print(f"Explaining target class: {args.target_class} ({prob_display})")

    # Compute gradients via backpropagation
    logit = logits[0, target_idx]
    model.zero_grad()
    logit.backward()

    # Saliency maps from gradients: shape (12, 1000) -> transpose to (1000, 12)
    gradients = X_torch.grad[0].cpu().numpy().transpose(1, 0)
    saliency = np.abs(gradients)

    # Smooth the saliency map for each lead
    saliency_smoothed = np.zeros_like(saliency)
    for c in range(12):
        saliency_smoothed[:, c] = gaussian_filter1d(saliency[:, c], sigma=args.smooth_sigma)

    # Normalize globally across all leads
    global_max = np.percentile(saliency_smoothed, 99)
    normalized_saliency = saliency_smoothed / (global_max + 1e-8)
    normalized_saliency = np.clip(normalized_saliency, 0.0, 1.0)

    # Prepare plotting
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time_sec = np.arange(1000) / fs

    # Layout setup: stacked plot
    # Offset value between leads in mV. Default 3.0 mV to prevent overlap.
    max_amp = np.max(np.abs(signal))
    offset = max(3.0, max_amp * 1.5)
    print(f"Lead vertical spacing offset: {offset:.2f} mV")

    fig, ax = plt.subplots(figsize=(15, 10))

    # Standard ECG paper grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))    # 200 ms major lines
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))   # 40 ms minor lines
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))    # 0.5 mV major lines
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))    # 0.1 mV minor lines

    # Grid aesthetics (pinkish/reddish grid matching real ECG graph paper)
    ax.grid(which='major', color='#ffcccc', linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)
    ax.grid(which='minor', color='#ffe6e6', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Prepend a 1 mV calibration pulse (200 ms width) for each lead baseline
    t_pulse = np.array([-0.4, -0.3, -0.3, -0.1, -0.1, 0.0])
    val_pulse = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    for i in range(12):
        y_offset = (11 - i) * offset
        
        # 1. Plot Calibration Pulse
        ax.plot(t_pulse, val_pulse + y_offset, color='black', linewidth=1.2, zorder=2)
        
        # 2. Plot clean ECG signal line
        lead_y = signal[:, i] + y_offset
        ax.plot(time_sec, lead_y, color='black', linewidth=1.1, zorder=2)
        
        # 3. Label lead on the left
        ax.text(-0.48, y_offset, lead_names[i], ha='right', va='center', fontweight='bold', fontsize=12)

        # 4. Overlay Saliency scatter points
        s_lead = normalized_saliency[:, i]
        mask = s_lead >= args.threshold
        if np.any(mask):
            cmap = plt.get_cmap(args.color_map)
            rgba_colors = cmap(s_lead)
            # Adjust individual transparency of scatter dots proportional to saliency score
            rgba_colors[:, 3] = 0.15 + 0.65 * s_lead
            
            ax.scatter(time_sec[mask], lead_y[mask], s=30 + 120 * s_lead[mask],
                       c=rgba_colors[mask], edgecolors='none', zorder=3)

    # Style axes
    ax.set_xlim(-0.6, 10.0)
    ax.set_ylim(-1.5, 11 * offset + 2.0)
    ax.set_xlabel("Time in seconds", fontweight='bold', labelpad=10)
    
    # Hide standard y-axis ticks/labels, keep grid
    ax.set_yticklabels([])
    
    # Clean up spines (ECG borders)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#7f8c8d')
    ax.spines['bottom'].set_linewidth(1.2)

    # Retrieve class description for the title
    if is_superclass:
        subclass_desc = statements.loc[args.target_class].description if args.target_class in statements.index else args.target_class
        title_text = f"ECG Saliency Attribution Map | Target: Superclass {superclass_name}\n" \
                     f"Explaining via Subclass: {args.target_class} ({subclass_desc}) | {prob_display}"
    else:
        if args.target_class in statements.index:
            desc = statements.loc[args.target_class].description
        else:
            desc = args.target_class
        title_text = f"ECG Saliency Attribution Map | Target: {args.target_class} ({desc})\n" \
                     f"{prob_display}"

    # Build title with metadata
    true_desc_list = []
    for code in true_scp_codes:
        if code in statements.index:
            true_desc_list.append(f"{code} ({statements.loc[code].description})")
        else:
            true_desc_list.append(code)
    true_labels_str = ", ".join(true_desc_list)
    title_text += f"\nTrue Diagnosis: {true_labels_str}"
    
    ax.set_title(title_text, pad=20, fontsize=13, fontweight='bold', color='#2c3e50')

    # Save output
    if args.output is None:
        target_name = superclass_name if is_superclass else args.target_class
        args.output = os.path.join(plots_dir, f"explain_{args.ecg_id}_{target_name}.png")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Successfully generated saliency map. Plot saved to: {args.output}")

if __name__ == "__main__":
    main()
