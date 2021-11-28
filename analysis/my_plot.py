
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
import numpy as np
# import copy

import matplotlib
from matplotlib import rc
# font = {'size'   : 16}
# matplotlib.rc('font', **font)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
# change font
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

from my_plot_data import MyPlotData

def compute_aspect_ratio(height, width, aspect):
    if height and width:
        aspect = None
    if height is None and width is None:
        height = 6
    if aspect is None:
        aspect = width/height
    if height is None:
        height = width/aspect
    print(f'Height: {height}, Aspect: {aspect}')
    return height, aspect


def my_box_plot(
    mpd,
    y,
    ylim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    kind='box',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    height=None,
    width=None,
    aspect=1/1.6,
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.4
    sns.set_style('whitegrid')
    sns.set_context(context, font_scale=font_scale)
    g = sns.catplot(
        kind=kind,
        y=y,
        data=mpd.to_dataframe(),
        linewidth=1,
        height=height, aspect=aspect,
        whis=(10, 90),
        )
    if ylim:
        g.ax.set_ylim(ylim)
    if y_tick_interval:
        lims = g.ax.get_ylim()
        g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    g.set_axis_labels(x_axis_label, y_axis_label)
    plt.tight_layout()
    if save_filename is None:
        plt.show()
    else:
        plt.savefig(save_filename, bbox_inches='tight')


# color = dict(boxes='black', whiskers='black', medians='red', caps='black')
# whiskerprops = dict(linestyle='-',linewidth=1, color='black')
# meanprops = dict(linestyle='-',linewidth=1, color='black')

def my_catplot(
    mpd,
    y,
    x=None,
    hue=None,
    kind='bar',
    hue_order=None,
    ylim=None,
    xlim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    add_swarm=False,
    custom_legend_loc=False,
    custom_legend_fn=None,
    close=True,
    linewidth=1,
    add_box=False,
    add_strip=False,
    add_kwargs={},
    add_data=None,
    **kwargs,
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    if context == 'paper':
        sns.set_style('ticks')
    else:
        sns.set_style('whitegrid')
    sns.set_context(context, font_scale=font_scale)
    g = sns.catplot(
        kind=kind,
        x=x, y=y, hue=hue,
        hue_order=hue_order,
        data=mpd.to_dataframe(),
        linewidth=linewidth,
        height=height, aspect=aspect,
        **kwargs,
        )
    if add_data is None:
        add_data = mpd
    if add_swarm:
        sns.swarmplot(
            ax=g.ax,
            x=x, y=y,
            # hue=hue,
            data=add_data.to_dataframe(),
            color=".25",
            )
    if add_box:
        sns.boxplot(
            ax=g.ax,
            x=x, y=y,
            data=add_data.to_dataframe(),
            )
    if add_strip:
        sns.stripplot(
            ax=g.ax,
            x=x, y=y,
            data=add_data.to_dataframe(),
            **add_kwargs,
            )
    if ylim:
        g.ax.set_ylim(ylim)
    if xlim:
        g.ax.set_xlim(xlim)
    if y_tick_interval:
        lims = g.ax.get_ylim()
        g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    # if kind != "violin":
    plt.tight_layout()
    if g.legend:
        g.legend.set_title("")
    g.set_axis_labels(x_axis_label, y_axis_label)
    if xticklabels:
        g.set_xticklabels(xticklabels)
    if custom_legend_fn:
        g._legend.remove()
        custom_legend_fn(plt)
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
        # plt.clf()
    # asdf
    if show or save_filename is None:
        # pass
        # asdf
        plt.show()
    else:
        if close:
            # asdf
            plt.close()
    return g


def my_cat_bar_plot(*args, **kwargs):
    return my_catplot(*args, **kwargs)


def my_displot(
    mpd,
    x,
    y=None,
    hue=None,
    style=None,
    size=None,
    alpha=1.0,
    kind='hist',
    hue_order=None,
    ylim=None,
    xlim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    palette=None,
    draw_lines=None,
    log_scale_x=False,
    custom_legend_fn=None,
    **kwargs
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    # sns.set_style('whitegrid')
    sns.set_context(context, font_scale=font_scale)
    if context == 'talk':
        sns.set_style('whitegrid')
    else:
        sns.set_style('ticks')
    g = sns.displot(
        # kind="point",
        kind=kind,
        x=x, y=y, hue=hue,
        # style=style,
        # size=size,
        # alpha=alpha,
        hue_order=hue_order,
        data=mpd.to_dataframe(),
        linewidth=1,
        height=height, aspect=aspect,
        # ci='sd',
        palette=palette,
        **kwargs
        # ci=5,
        # whis=(10, 90),
        )

    if log_scale_x:
        for ax in g.axes[0]:
            ax.set(xscale='log')

    if draw_lines:
        print(g.axes)
        print(g.axes[0]) 
        print(g.axes_dict)
        my_draw_lines(draw_lines, g.axes[0][0])

    if ylim:
        g.ax.set_ylim(ylim)
    if xlim:
        g.ax.set_xlim(xlim)
    if y_tick_interval:
        lims = g.ax.get_ylim()
        g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    plt.tight_layout()
    if g.legend:
        g.legend.set_title("")
    g.set_axis_labels(x_axis_label, y_axis_label)
    if xticklabels:
        g.set_xticklabels(xticklabels)
    if custom_legend_fn:
        g._legend.remove()
        custom_legend_fn(plt)
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
        # plt.clf()
    if show or save_filename is None:
        plt.show()
    else:
        plt.close()



def my_relplot(
    mpd,
    x, y, hue=None,
    size=None,
    alpha=1.0,
    kind='line',
    hue_order=None,
    ylim=None,
    xlim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    palette=None,
    draw_lines=None,
    custom_legend_loc=False,
    custom_legend_fn=None,
    log_scale_x=False,
    log_scale_y=False,
    xticks=None,
    title=None,
    old_tight_layout=False,
    tight_layout=False,
    linewidth=1,
    **kwargs
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    if context == 'talk':
        sns.set_style('whitegrid')
    else:
        sns.set_style('ticks')
    sns.set_context(context, font_scale=font_scale)
    g = sns.relplot(
        # kind="point",
        kind=kind,
        x=x, y=y, hue=hue,
        size=size,
        alpha=alpha,
        hue_order=hue_order,
        data=mpd.to_dataframe(),
        linewidth=linewidth,
        height=height, aspect=aspect,
        # ci='sd',
        palette=palette,
        **kwargs
        # ci=5,
        # whis=(10, 90),
        )

    if draw_lines:
        print(g.axes)
        print(g.axes[0]) 
        print(g.axes_dict)
        my_draw_lines(draw_lines, g.axes[0][0])
    if xlim:
        g.ax.set_xlim(xlim)
    if log_scale_x:
        for ax in g.axes[0]:
            ax.set(xscale='log')
    if log_scale_y:
        for ax in g.axes[0]:
            ax.set(yscale='log')
    if ylim:
        g.ax.set_ylim(ylim)
    if y_tick_interval:
        lims = g.ax.get_ylim()
        g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    if xticks:
        for ax in g.axes[0]:
            ax.set_xticks(xticks)
    # g.set_axis_labels(x_axis_label, y_axis_label, fontname='Arial')
    # g.set_axis_labels(x_axis_label, y_axis_label, fontname='Monospace')
    g.set_axis_labels(x_axis_label, y_axis_label)
    if xticklabels:
        g.set_xticklabels(xticklabels)
    if title:
        for ax in g.axes[0]:
            ax.set_title(title)
    if tight_layout:
        plt.tight_layout()
    if custom_legend_loc:
        g._legend.remove()
        plt.legend(loc=custom_legend_loc)
    if custom_legend_fn:
        g._legend.remove()
        custom_legend_fn(plt)
    if g.legend:
        g.legend.set_title("")
    if old_tight_layout:
        plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
        # plt.clf()
    if show or save_filename is None:
        plt.show()
    else:
        plt.close()
    return g



def my_regplot(
    mpd,
    x, y,
    # hue=None,
    # style=None,
    # size=None,
    # alpha=1.0,
    # hue_order=None,
    ylim=None,
    xlim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    # palette=None,
    # draw_lines=None,
    custom_legend_loc=False,
    custom_legend_fn=None,
    log_scale_x=False,
    log_scale_y=False,
    xticks=None,
    title=None,
    old_tight_layout=False,
    tight_layout=False,
    # linewidth=1,
    **kwargs
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    sns.set_context(context, font_scale=font_scale)
    if context == 'talk':
        sns.set_style('whitegrid')
    else:
        sns.set_style('ticks')
    g = sns.regplot(
    # g = sns.lmplot(
        x=x, y=y,
        # hue=hue,
        # style=style,
        # size=size,
        # alpha=alpha,
        # hue_order=hue_order,
        data=mpd.to_dataframe(),
        # linewidth=linewidth,
        # height=height, aspect=aspect,
        # palette=palette,
        **kwargs
        )

    # return g

    if log_scale_x:
        for ax in g.axes[0]:
            ax.set(xscale='log')
    if log_scale_y:
        for ax in g.axes[0]:
            ax.set(yscale='log')
    if ylim:
        g.axes.set_ylim(ylim)
    if xlim:
        g.axes.set_xlim(xlim)
    if y_tick_interval:
        lims = g.axes.get_ylim()
        g.axes.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    g.axes.set_xlabel(x_axis_label)
    g.axes.set_ylabel(y_axis_label)
    if xticks:
        g.axes.set_xticks(xticks)
    if xticklabels:
        g.axes.set_xticklabels(xticklabels)
    # if title:
    #     for ax in g.axes[0]:
    #         ax.set_title(title)
    if tight_layout:
        plt.tight_layout()
    if custom_legend_loc:
        g._legend.remove()
        plt.legend(loc=custom_legend_loc)
    if custom_legend_fn:
        g._legend.remove()
        custom_legend_fn(plt)
    # if g.legend:
    #     g.legend.set_title("")
    # if old_tight_layout:
    #     plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
    if show or save_filename is None:
        plt.show()
    else:
        plt.close()
    return g


def my_lmplot(
    mpd,
    x, y,
    # hue=None,
    # style=None,
    # size=None,
    # alpha=1.0,
    # hue_order=None,
    ylim=None,
    xlim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    # palette=None,
    # draw_lines=None,
    custom_legend_loc=False,
    custom_legend_fn=None,
    log_scale_x=False,
    log_scale_y=False,
    xticks=None,
    title=None,
    old_tight_layout=False,
    tight_layout=False,
    # linewidth=1,
    **kwargs
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    sns.set_context(context, font_scale=font_scale)
    if context == 'talk':
        sns.set_style('whitegrid')
    else:
        sns.set_style('ticks')
    g = sns.lmplot(
        x=x, y=y,
        # hue=hue,
        # style=style,
        # size=size,
        # alpha=alpha,
        # hue_order=hue_order,
        data=mpd.to_dataframe(),
        # linewidth=linewidth,
        height=height, aspect=aspect,
        # palette=palette,
        **kwargs
        )

    # return g

    if log_scale_x:
        for ax in g.axes[0]:
            ax.set(xscale='log')
    if log_scale_y:
        for ax in g.axes[0]:
            ax.set(yscale='log')
    if ylim:
        g.ax.set_ylim(ylim)
    if xlim:
        g.ax.set_xlim(xlim)
    if y_tick_interval:
        lims = g.ax.get_ylim()
        g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    g.ax.set_xlabel(x_axis_label)
    g.ax.set_ylabel(y_axis_label)
    if xticks:
        g.ax.set_xticks(xticks)
    if xticklabels:
        g.ax.set_xticklabels(xticklabels)
    # if title:
    #     for ax in g.axes[0]:
    #         ax.set_title(title)
    if tight_layout:
        plt.tight_layout()
    if custom_legend_loc:
        g._legend.remove()
        plt.legend(loc=custom_legend_loc)
    if custom_legend_fn:
        g._legend.remove()
        custom_legend_fn(plt)
    # if g.legend:
    #     g.legend.set_title("")
    # if old_tight_layout:
    #     plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
    if show or save_filename is None:
        plt.show()
    else:
        plt.close()
    return g


def my_lineplot(
    mpd,
    x, y, hue=None,
    style=None,
    size=None,
    alpha=1.0,
    hue_order=None,
    ylim=None,
    xlim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    palette=None,
    draw_lines=None,
    custom_legend_loc=False,
    custom_legend_fn=None,
    log_scale_x=False,
    log_scale_y=False,
    xticks=None,
    title=None,
    old_tight_layout=False,
    tight_layout=False,
    linewidth=1,
    no_show=False,
    **kwargs
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    sns.set_context(context, font_scale=font_scale)
    sns.set_style('ticks')
    g = sns.lineplot(
        # kind="point",
        x=x, y=y, hue=hue,
        style=style,
        size=size,
        alpha=alpha,
        hue_order=hue_order,
        data=mpd.to_dataframe(),
        linewidth=linewidth,
        # height=height, aspect=aspect,
        # ci='sd',
        palette=palette,
        legend=False,
        **kwargs
        # ci=5,
        # whis=(10, 90),
        )

    if draw_lines:
        print(g.axes)
        print(g.axes[0]) 
        print(g.axes_dict)
        my_draw_lines(draw_lines, g.axes[0][0])
    if log_scale_x:
        for ax in g.axes[0]:
            ax.set(xscale='log')
    if log_scale_y:
        for ax in g.axes[0]:
            ax.set(yscale='log')
    if ylim:
        g.set_ylim(ylim)
    if xlim:
        g.set_xlim(xlim)
    if y_tick_interval:
        lims = g.ax.get_ylim()
        g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    if xticks:
        for ax in g.axes[0]:
            ax.set_xticks(xticks)
    # g.set_axis_labels(x_axis_label, y_axis_label)
    g.set_xlabel(x_axis_label)
    g.set_ylabel(y_axis_label)
    if xticklabels:
        g.set_xticklabels(xticklabels)
    if title:
        for ax in g.axes[0]:
            ax.set_title(title)
    if tight_layout:
        plt.tight_layout()
    if custom_legend_loc:
        g._legend.remove()
        plt.legend(loc=custom_legend_loc)
    if custom_legend_fn:
        g._legend.remove()
        custom_legend_fn(plt)
    # if g.legend:
    #     g.legend.set_title("")
    # g.legend()
    if old_tight_layout:
        plt.tight_layout()

    if not no_show:
        if save_filename:
            plt.savefig(save_filename, bbox_inches='tight', transparent=True)
            # plt.clf()
        if show or save_filename is None:
            plt.show()
        else:
            plt.close()
    return g



def my_jointplot(
    mpd,
    x, y,
    hue=None,
    kind='scatter',
    hue_order=None,
    xlim=None,
    ylim=None,
    y_tick_interval=None,
    save_filename=None,
    context='paper',
    font_scale=None,
    x_axis_label='',
    y_axis_label='',
    show=False,
    xticklabels=None,
    height=None,
    width=None,
    aspect=1.33,
    draw_lines=None,
    log_scale_x=None,
    log_scale_y=None,
    **kwargs
    ):
    height, aspect = compute_aspect_ratio(height, width, aspect)
    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    if context == 'paper':
        sns.set_style('ticks')
    else:
        sns.set_style('whitegrid')
    sns.set_context(context, font_scale=font_scale)
    g = sns.jointplot(
        # kind="point",
        kind=kind,
        x=x, y=y, 
        hue=hue,
        hue_order=hue_order,
        data=mpd.to_dataframe(),
        # linewidth=1,
        height=height,
        # aspect=aspect,
        # ci='sd',
        # ci=5,
        xlim=xlim,
        ylim=ylim,
        # whis=(10, 90),
        **kwargs,
        )

    if draw_lines:
        my_draw_lines(my_draw_lines, g.axes[0])

    if log_scale_x:
        g.ax_joint.set_xscale('log')
    if log_scale_y:
        g.ax_joint.set_yscale('log')
    # max_lim = max(g.ax.get_ylim()[1], g.ax.get_xlim()[1])
    # g.ax.set_ylim((g.ax.get_ylim()[0], max_lim))
    # g.ax.set_xlim((g.ax.get_xlim()[0], max_lim))
    # if ylim:
    #     g.ax.set_ylim(ylim)
    # if y_tick_interval:
    #     lims = g.ax.get_ylim()
    #     g.ax.set_yticks(np.arange(lims[0], lims[1]+0.001, y_tick_interval))
    plt.tight_layout()
    # g.legend.set_title("")
    g.set_axis_labels(x_axis_label, y_axis_label)
    if xticklabels:
        g.set_xticklabels(xticklabels)
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
    if show or save_filename is None:
        plt.show()
    else:
        plt.close()


def my_draw_lines(
        point_pairs,
        ax=None
        ):
    if ax is None:
        ax = plt.gca()

    for pair in point_pairs:
        print(f'Plotting from {pair[0]} to {pair[1]}')
        ax.plot(
            [pair[0][0], pair[1][0]],
            [pair[0][1], pair[1][1]],
        # ax.plot(point_pairs,
            linestyle='--', linewidth=0.75, color='grey')


def my_scatterplot(
        mpd,
        ax=None,
        xlim=None,
        ylim=None,
        x_axis_label=None,
        y_axis_label=None,
        # height=None,
        # width=None,

        kind='scatter',

        context='paper',
        font_scale=None,

        save_filename=None,

        **kwargs,
        ):
    if ax is None:
        ax = plt.gca()

    if font_scale is None:
        if context == 'talk': font_scale = 1
        if context == 'paper': font_scale = 1.5
    if context == 'paper':
        sns.set_style('ticks')
        # sns.set_style('whitegrid')
    else:
        sns.set_style('whitegrid')
    sns.set_context(context, font_scale=font_scale)

    if kind == 'scatter':
        sns.scatterplot(
            data=mpd.to_dataframe(),
            ax=ax,
            **kwargs,
            )
    elif kind == 'kde':
        sns.kdeplot(
            data=mpd.to_dataframe(),
            ax=ax,
            **kwargs,
            )

    if xlim:
        ax.axes.set_xlim(xlim)
    if ylim:
        ax.axes.set_ylim(ylim)
    if x_axis_label:
        ax.axes.set_xlabel(x_axis_label)
    if y_axis_label:
        ax.axes.set_ylabel(y_axis_label)

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
    # if close:
        # plt.close()
    return ax



