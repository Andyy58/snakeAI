import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # enable interactivity


def plot(
    scores, mean_scores
):  # Plot scores and mean scores; plot_scores as x-axis, plot_mean_scores as y-axis
    display.clear_output(wait=True)  # Clear output
    display.display(plt.gcf())  # Get current figure
    plt.clf()  # Clear figure
    plt.title("Training...")  # Title
    plt.xlabel("Number of Games")  # x-axis label
    plt.ylabel("Score")  # y-axis label
    plt.plot(scores)  # Plot scores
    plt.plot(mean_scores)  # Plot mean scores
    plt.ylim(ymin=0)  # Set y-axis minimum to 0
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # Set text for last score
    plt.text(
        len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1])
    )  # Set text for last mean score
    plt.show(block=False)  # Show plot
    plt.pause(0.1)  # Pause a bit so that plots are updated
