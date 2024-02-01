import json
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns


def plot_metric(data, metric_name, x_label, y_label, title, colors, max_epoch=30):
    # plt.figure(figsize=(7, 6))
    # Create single plot.
    fig, ax = plt.subplots(figsize=(413 / 40, 238 / 40))
    # Create twin y axis.
    ax2 = ax.twinx()

    dim = range(1, max_epoch + 1)

    # Enable grid.
    ax.grid(True)

    train = ax.plot(
        dim,
        data[f"train_{metric_name}"][:max_epoch],
        label=f"Train {metric_name.capitalize()}",
        color=colors[0],
        lw=3,
    )
    val = ax.plot(
        dim,
        data[f"val_{metric_name}"][:max_epoch],
        label=f"Validation {metric_name.capitalize()}",
        color=colors[1],
        lw=3,
    )
    alpaca = ax.plot(
        dim,
        data[f"alpaca_{metric_name}"][:max_epoch],
        label=f"Validation {metric_name.capitalize()} 'Alpaca'",
        color=colors[2],
        lw=3,
    )
    # Smooth rhyming percentage.
    rhyming_data = data[f"rhyming"][:max_epoch]
    rhyming_data_smooth = []
    for i in range(len(rhyming_data)):
        if i < 10:
            rhyming_data_smooth.append(rhyming_data[i])
        else:
            rhyming_data_smooth.append(sum(rhyming_data[i - 5 : i]) / 5)

    rhyming = ax2.plot(
        dim,
        rhyming_data_smooth,
        label=f"Rhyming Percentage",
        color=colors[3],
        linestyle="--",
        lw=3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax2.set_ylabel("Rhyming Percentage")
    ax.set_title(f"Validation Metrics")
    # Collect all plots for common legend.
    plots = train + val + alpaca + rhyming
    # Collect all labels for common legend.
    labels = [l.get_label() for l in plots]
    # Create common legend.
    ax.legend(plots, labels, loc=0)
    fig.tight_layout()


def plot_single_metric_by_step(
    data, metric_name, x_label, y_label, title, color, max_epoch
):
    plt.plot(data[f"{metric_name}"][:max_epoch], label=f"{title}", color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_metrics_by_step(data, metric_name, x_label, y_label, colors, max_epoch):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plot_single_metric_by_step(
        data,
        f"train_step_{metric_name}",
        x_label,
        y_label,
        f"Train Step {metric_name.capitalize()}",
        colors[0],
        max_epoch,
    )
    plt.subplot(1, 3, 2)
    plot_single_metric_by_step(
        data,
        f"val_step_{metric_name}",
        x_label,
        y_label,
        f"Validation Step {metric_name.capitalize()}",
        colors[1],
        max_epoch,
    )
    plt.subplot(1, 3, 3)
    plot_single_metric_by_step(
        data,
        f"alpaca_step_{metric_name}",
        x_label,
        y_label,
        f"Alpaca Validation Step {metric_name.capitalize()}",
        colors[2],
        max_epoch,
    )
    plt.tight_layout()


def plot_metrics(file_path, max_epoch=10):
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
    # rojo = "#DD1C1A"
    # plum = 9C528B
    colors = ["#588b8b", "#ab9e93", "#9C528B", "#1E1014"]

    plot_metric(data, "loss", "Epoch", "Loss", "Loss", colors, max_epoch)
    plt.savefig(
        os.path.join(directory, f"{filename_prefix}_train_and_validation_loss.png"),
        format="png",
        transparent=True,
    )
    plt.close()

    plot_metric(
        data, "perplexity", "Epoch", "Perplexity", "Perplexity", colors, max_epoch
    )
    plt.savefig(
        os.path.join(
            directory, f"{filename_prefix}_train_and_validation_perplexity.png"
        ),
        format="png",
        transparent=True,
    )
    plt.close()

    plot_metrics_by_step(data, "loss", "Step", "Loss", colors, max_epoch)
    plt.savefig(
        os.path.join(
            directory, f"{filename_prefix}_train_and_validation_loss_by_step.png"
        ),
        format="png",
        transparent=True,
    )
    plt.close()

    plot_metrics_by_step(data, "perplexity", "Step", "Perplexity", colors, max_epoch)
    plt.savefig(
        os.path.join(
            directory, f"{filename_prefix}_train_and_validation_perplexity_by_step.png"
        ),
        format="png",
        transparent=True,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from JSON file.")
    parser.add_argument(
        "--file_path", required=True, type=str, help="Path to the metrics JSON file."
    )
    # Number of epochs to plot.
    parser.add_argument(
        "--max_epoch",
        required=False,
        type=int,
        default=30,
        help="Number of epochs to plot.",
    )
    args = parser.parse_args()
    # Update plt font size.
    plt.rcParams.update({"font.size": 13})
    plot_metrics(args.file_path, args.max_epoch)
