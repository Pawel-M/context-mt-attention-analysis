import matplotlib.pyplot as plt
import numpy as np


def image_plot(values, x_ticks=None, y_ticks=None, x_label=None, y_label=None, rotate_x_ticks=True, title=None,
               fig_size=None,
               save_path=None, show=True,
               dpi=300, value_string='{:1.1f}',
               cmap='viridis', reverse_cmap=False,
               increase_upward=False):
    if type(values) == list:
        values_size = (len(values), values[0].shape[0])
    else:
        values_size = values.shape

    cmap = plt.get_cmap(cmap + ('_r' if reverse_cmap else ''))
    origin = 'lower' if increase_upward else 'upper'

    fig, ax = plt.subplots()
    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    im = ax.imshow(values, cmap=cmap, origin=origin)
    if x_ticks is not None:
        ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
    if y_ticks is not None:
        ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if rotate_x_ticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if value_string is not None:
        for i in range(values_size[1]):
            for j in range(values_size[0]):
                text = ax.text(i, j, value_string.format(values[j][i]), ha="center", va="center", color="w")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()

    plt.close()


def plot_attention(attentions, src_tokens, tgt_tokens, correct, show=True, save_file=None, value_precision=1):
    cell_size = 0.3
    fig_height = max(len(src_tokens) * cell_size + 2, 5)
    fig_width = max(len(tgt_tokens) * cell_size + 2, 5)
    image_plot(attentions.detach().numpy().T,
               title=f'Correct: {correct}',
               x_ticks=tgt_tokens,
               y_ticks=src_tokens,
               x_label='target',
               y_label='source',
               rotate_x_ticks=True,
               fig_size=(fig_width, fig_height),
               show=show,
               save_path=save_file, value_string=f'{{:1.{value_precision}f}}')


