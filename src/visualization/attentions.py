import os

import torch
import tqdm

from visualization.plot_utils import plot_attention


def plot_attentions(results_dir, correct,
                    self_attentions, cross_attentions, decoder_attentions, tokens,
                    limit_plots, plot_separate_attentions, plot_separate_heads, results_suffix=None):
    if limit_plots is not None and limit_plots < 1:
        return

    if results_suffix is not None and len(results_suffix) > 0:
        dir_name = f'cross_attentions_{results_suffix}'
    else:
        dir_name = 'cross_attentions'

    plots_dir = os.path.join(results_dir, dir_name)
    os.makedirs(plots_dir, exist_ok=True)

    p_self_attentions = self_attentions
    p_cross_attentions = cross_attentions
    p_decoder_attentions = decoder_attentions
    p_tokens = tokens
    p_correct = correct

    if limit_plots is not None:
        p_self_attentions = p_self_attentions[:limit_plots]
        p_cross_attentions = p_cross_attentions[:limit_plots]
        p_decoder_attentions = p_decoder_attentions[:limit_plots]
        p_tokens = p_tokens[:limit_plots]
        p_correct = p_correct[:limit_plots]

    print(f'Generating {len(p_tokens)} plots...')

    i = 0
    pbar = tqdm.tqdm(zip(p_self_attentions, p_cross_attentions, p_decoder_attentions, p_tokens, p_correct),
                     dynamic_ncols=True, ncols=200)
    for self_attention, cross_attention, decoder_attention, ts, c in pbar:
        self_attentions_plot_file = os.path.join(plots_dir, f'self_attention_{i}')
        cross_attentions_plot_file = os.path.join(plots_dir, f'cross_attention_{i}')
        decoder_attentions_plot_file = os.path.join(plots_dir, f'decoder_attention_{i}')

        sast = torch.stack(self_attention)
        avg_sa = torch.max(sast[:, 0, ...], dim=0)[0]
        avg_sa = torch.max(avg_sa, dim=0)[0]
        plot_attention(avg_sa, ts[0], ts[0], c, show=False, save_file=self_attentions_plot_file + '.png')

        cast = torch.stack(cross_attention[0])
        avg_cas = torch.max(cast, dim=0)[0][0]
        avg_cas = torch.max(avg_cas, dim=0)[0]
        plot_attention(avg_cas, ts[0], ts[1][0], c, show=False, save_file=cross_attentions_plot_file + '.png')

        dast = torch.stack(decoder_attention[0])
        avg_das = torch.max(dast, dim=0)[0][0]
        avg_das = torch.max(avg_das, dim=0)[0]
        plot_attention(avg_das, ts[1][0], ts[1][0], c, show=False, save_file=decoder_attentions_plot_file + '.png')

        if plot_separate_attentions:
            for layer_index, (self_attn, cross_attn, dec_attn) in enumerate(
                    zip(self_attention, cross_attention[0], decoder_attention[0])):
                avg_sa = torch.max(self_attn[0, ...], dim=0)[0]
                plot_file = self_attentions_plot_file + f'_{(layer_index + 1)}'
                plot_attention(avg_sa, ts[0], ts[0], c, show=False, save_file=plot_file + '.png')

                avg_ca = torch.max(cross_attn[0, ...], dim=0)[0]
                plot_file = cross_attentions_plot_file + f'_{(layer_index + 1)}'
                plot_attention(avg_ca, ts[0], ts[1][0], c, show=False, save_file=plot_file + '.png')

                avg_da = torch.max(dec_attn[0, ...], dim=0)[0]
                plot_file = decoder_attentions_plot_file + f'_{(layer_index + 1)}'
                plot_attention(avg_da, ts[1][0], ts[1][0], c, show=False, save_file=plot_file + '.png')

                if plot_separate_heads:
                    for head_index in range(self_attn.shape[1]):
                        plot_file_suffix = f'_{(layer_index + 1)}_{(head_index + 1)}'
                        plot_attention(self_attn[0, head_index], ts[0], ts[0], c, show=False,
                                       save_file=self_attentions_plot_file + plot_file_suffix + '.png',
                                       value_precision=2)
                        plot_attention(cross_attn[0, head_index], ts[0], ts[1][0], c, show=False,
                                       save_file=cross_attentions_plot_file + plot_file_suffix + '.png',
                                       value_precision=2)
                        plot_attention(dec_attn[0, head_index], ts[1][0], ts[1][0], c, show=False,
                                       save_file=decoder_attentions_plot_file + plot_file_suffix + '.png',
                                       value_precision=2)

        i += 1
