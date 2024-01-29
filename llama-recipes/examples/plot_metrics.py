import json
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns


def plot_metric(data, metric_name, x_label, y_label, title, colors, max_epoch=10):
    plt.figure(figsize=(7, 6))

    plt.plot(
        data[f"train_{metric_name}"][:max_epoch],
        label=f"Train Epoch {metric_name.capitalize()}",
        color=colors[0],
    )
    plt.plot(
        data[f"val_{metric_name}"][:max_epoch],
        label=f"Validation Epoch {metric_name.capitalize()}",
        color=colors[1],
    )
    plt.plot(
        data[f"alpaca_{metric_name}"][:max_epoch],
        label=f"Alpaca Validation Epoch {metric_name.capitalize()}",
        color=colors[2],
    )
    plt.plot(
        data[f"rhyming"][:max_epoch],
        label=f"Rhyming Percentage",
        color=colors[3],
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Train and Validation Epoch {title}")
    plt.legend()
    plt.tight_layout()


def plot_single_metric_by_step(data, metric_name, x_label, y_label, title, color):
    plt.plot(data[f"{metric_name}"], label=f"{title}", color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_metrics_by_step(data, metric_name, x_label, y_label, colors):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plot_single_metric_by_step(
        data,
        f"train_step_{metric_name}",
        x_label,
        y_label,
        f"Train Step {metric_name.capitalize()}",
        colors[0],
    )
    plt.subplot(1, 3, 2)
    plot_single_metric_by_step(
        data,
        f"val_step_{metric_name}",
        x_label,
        y_label,
        f"Validation Step {metric_name.capitalize()}",
        colors[1],
    )
    plt.subplot(1, 3, 3)
    plot_single_metric_by_step(
        data,
        f"alpaca_step_{metric_name}",
        x_label,
        y_label,
        f"Alpaca Validation Step {metric_name.capitalize()}",
        colors[2],
    )
    plt.tight_layout()


def plot_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Invalid JSON file.")
            return

    directory = os.path.dirname(file_path)
    filename_prefix = os.path.basename(file_path).split(".")[0]
    number_of_metrics = 4
    colors = sns.color_palette("crest", number_of_metrics)

    plot_metric(data, "loss", "Epoch", "Loss", "Loss", colors)
    plt.savefig(
        os.path.join(directory, f"{filename_prefix}_train_and_validation_loss.png")
    )
    plt.close()

    plot_metric(data, "perplexity", "Epoch", "Perplexity", "Perplexity", colors)
    plt.savefig(
        os.path.join(
            directory, f"{filename_prefix}_train_and_validation_perplexity.png"
        )
    )
    plt.close()

    plot_metrics_by_step(data, "loss", "Step", "Loss", colors)
    plt.savefig(
        os.path.join(
            directory, f"{filename_prefix}_train_and_validation_loss_by_step.png"
        )
    )
    plt.close()

    plot_metrics_by_step(data, "perplexity", "Step", "Perplexity", colors)
    plt.savefig(
        os.path.join(
            directory, f"{filename_prefix}_train_and_validation_perplexity_by_step.png"
        )
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from JSON file.")
    parser.add_argument(
        "--file_path", required=True, type=str, help="Path to the metrics JSON file."
    )
    args = parser.parse_args()

    plot_metrics(args.file_path)
