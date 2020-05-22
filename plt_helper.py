import re
import pandas
import matplotlib.pyplot as plt
import numpy as np


def title_to_filename(title):
    return re.sub('[^0-9a-zA-Z]+', '_', title) + ".png"


def plot_histogram_distinct(title: str, data: np.ndarray, sorted=True):
    plt.title(title)
    if sorted:
        series = pandas.Series(data).value_counts()
    else:
        series = pandas.Series(data).value_counts().sort_index()
    series.plot(kind='bar')
    save_and_show_plot(title)


def plot_day_histogram(title: str, data):
    data = data.value_counts().sort_index()
    data.index = ['Montag', 'Dienstag', 'Mittwoch', 'Donerstag', 'Freitag', 'Samstag', 'Sonntag']
    data.plot(kind='bar')
    save_and_show_plot(title)


def save_and_show_plot(title: str, print_title=True):
    if print_title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig("plots/" + title_to_filename(title))
    plt.show()
