import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_loss_curve(file: str, save_path: str):
    """
    Plot the losses and save to image
    """
    available_priorities = ["BERT-base", "uniform", "entropy", "margin", "bald", "variance"]
    matched_priority = next((p for p in available_priorities if p in save_path), None)
    if matched_priority:
        title = f"Training Loss over steps using {matched_priority}"
    else:
        print(
            "Error in plot_loss_curve, save_path must contain"
            "one of the available priority functions"
        )
        return
    df = pd.read_csv(file)
    plt.figure(figsize=(8, 5))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    df.mean(axis=1).plot(title=title)
    plt.savefig(save_path)
    print(f"✅ Loss curve saved to {save_path}")

def plot_loss_and_accuracy_curve(loss_file: str, acc_file: str, save_path: str):
    """
    Plot the training loss and accuracy curves from a CSV file.
    """
    available_priorities = ["BERT-base", "uniform", "entropy", "margin", "bald", "variance"]
    matched_priority = next((p for p in available_priorities if p in save_path), None)

    if matched_priority:
        title = f"Training Metrics over Steps using {matched_priority}"
    else:
        print(
            "Error in plot_loss_and_accuracy_curve: save_path must contain "
            "one of the available priority functions"
        )
        return

    loss_df = pd.read_csv(loss_file)
    acc_df = pd.read_csv(acc_file)
    plt.figure(figsize=(10, 6))

    # Plot loss
    plt.plot(loss_df.mean(axis=1), label="Loss")
    # Plot accuracy
    plt.plot(acc_df.mean(axis=1), label="Accuracy")

    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Loss and accuracy curve saved to {save_path}")


def plot_all_runs_with_mean(loss_file: str, acc_file: str, save_path: str):
    """
    Plot all individual runs and their mean for loss and accuracy.
    """
#    available_priorities = ["ENN", "BERT-base", "uniform", "entropy", "margin", "bald", "variance"]
#    matched_priority = next((p for p in available_priorities if p in save_path), None)
#
#    if matched_priority:
#        title = f"Training Metrics over Steps using {matched_priority}"
#    else:
#        print(
#            "Error in plot_all_runs_with_mean: save_path must contain "
#            "one of the available priority functions"
#        )
#        return

    # Read data
    loss_df = pd.read_csv(loss_file)
    acc_df = pd.read_csv(acc_file)

    steps = loss_df['step'] if 'step' in loss_df.columns else loss_df.index

    plt.figure(figsize=(12, 7))

    # Plot each run for loss
    for col in loss_df.columns:
        if col != 'step':
            plt.plot(steps, loss_df[col], color='blue', alpha=0.3, linewidth=1)

    # Plot mean loss
    mean_loss = loss_df.drop(columns=['step']).mean(axis=1)
    plt.plot(steps, mean_loss, color='blue', label='Mean Loss', linewidth=3)

    # Plot each run for accuracy
    for col in acc_df.columns:
        if col != 'step':
            plt.plot(steps, acc_df[col], color='orange', alpha=0.3, linewidth=1)

    # Plot mean accuracy
    mean_acc = acc_df.drop(columns=['step']).mean(axis=1)
    plt.plot(steps, mean_acc, color='orange', label='Mean Accuracy', linewidth=3)

    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    #plt.title(title)
    plt.title("Training metrics over steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ All runs and mean saved to {save_path}")



def save_results_to_file(losses: list, accs: list):
    """
    Save the list of losses to a file.

    Args:
    losses (list): List of loss values.
    accs (list): List of accuracy values
    save_path (str): Path to the file where losses should be saved.
    """
    accs_save_path = "./accs.csv"
    losses_save_path = "./losses.csv"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(accs_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(losses_save_path), exist_ok=True)

    # Write losses to the file
    with open(losses_save_path, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    with open(accs_save_path, "w") as f:
        for acc in accs:
            f.write(f"{acc}\n")

    print(f"Losses saved to {losses_save_path}")
    print(f"Accs saved to {accs_save_path}")
