import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import ttest_rel, sem
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
import pingouin as pg
from pymer4.models import Lmer

from psyfun import io, util, atlas, spikes, plots
from psyfun.config import *

df_spikes = io.load_session_spikes()
df_spikes['coarse_region'] = atlas.coarse_regions(df_spikes['region'])

df_spikes['full_start'] = df_spikes[
    [col for col in df_spikes.columns if '_start' in col]
    ].min(axis='columns')
df_spikes['full_stop'] = df_spikes[
    [col for col in df_spikes.columns if '_stop' in col]
    ].max(axis='columns')

epochs = [
    'task00_spontaneous',
    'task00_replay',
    'task01_spontaneous',
    'task01_replay',
    # ~'full'
]

# Get spike counts for epochs
dt = 0.1  # seconds
for epoch in epochs:
    print(f"Getting spike counts for: {epoch}")
    # Convert spike times to spike counts in bins of size dt
    df_spikes[f'{epoch}_counts'] = df_spikes.progress_apply(
        lambda neuron:
            spikes._apply_spike_counts(
                neuron, epoch=epoch, dt=dt
                ),
        axis='columns'
        )

# Define population of neurons to loop over
populations = df_spikes.groupby('eid')
min_n = 100
print(populations.size())
populations = populations.filter(lambda x: len(x) > min_n).groupby('eid')

# Define which columns from the dataframe you want to keep for downstream analysis
columns_to_keep = ['subject', 'eid', 'session_n', 'control_recording', 'start_time', 'trajectory_label']
# columns_to_keep = ['subject', 'eid', 'corase_region', 'session_n', 'control_recording', 'start_time']

# Perform PCA on each population of neurons
pca_results = []  # initialize lis to stare PCA results
for idx, neurons in tqdm(populations, total=len(populations)):

    # Initialize a dict to store metadata and PCA results for this session
    population_data = {col: neurons[col].unique()[0] for col in columns_to_keep}

    # Perform PCA separately for spike couts from each epoch
    for epoch in epochs:
        # Put spike counts for all neurons from this recording into a samples x features matrix
        X = np.column_stack(neurons[f'{epoch}_counts'])

        # Normalize the data
        # ~scaler = StandardScaler(with_std=False)  # note: with_std set to false preserves rate variability while controlling for differences in firing rate across neurons
        scaler = StandardScaler()
        X_rescaled = scaler.fit_transform(X)
        # Alternative pre-processing step to satisfy gaussianity assumption in PCA (non-linear transformation to make spike counts normally distibuted within neurons)
        # transformer = PowerTransformer(method='yeo-johnson', standardize=True)  # note: standardization/ rescaling is included
        # X_rescaled = transformer.fit_transform(X)

        # Fit the PCA
        pca = PCA()
        pca.fit(X_rescaled)

        # Add explained variance and loadings to the data dict
        population_data[f'{epoch}_eigenspectrum'] = pca.explained_variance_ratio_
        population_data[f'{epoch}_loadings'] = pca.components_

    # Append results dict to list
    pca_results.append(population_data)

# Convert list of dicts to dataframe
df_pca = pd.DataFrame(pca_results)

for epoch in epochs:
    df_pca = df_pca.assign(**{  # note: using the assign method like this prevents datafram fragmentation
        f'{epoch}_pc1': df_pca[f'{epoch}_eigenspectrum'].apply(lambda x: x[0]),
        f'{epoch}_n80': df_pca[f'{epoch}_eigenspectrum'].apply(lambda x: np.cumsum(x).searchsorted(0.8)),
        f'{epoch}_powlaw': df_pca[f'{epoch}_eigenspectrum'].apply(util.power_law_slope),
        f'{epoch}_ngsc': df_pca[f'{epoch}_eigenspectrum'].apply(util.normalized_entropy)
    })

metrics = ['pc1', 'n80', 'powlaw', 'ngsc']
for metric in metrics:

    metric_cols = [
        col for col in df_pca.columns if metric in col
        ]
    df_metric = df_pca.melt(
        id_vars=['subject', 'eid', 'control_recording'],
        value_vars=metric_cols,
        var_name='epoch',
        value_name=metric
        )
    df_metric['task'] = df_metric['epoch'].apply(
        lambda x: x.split('_')[0]
        ).astype('category')
    df_metric['epoch'] = df_metric['epoch'].apply(
        lambda x: x.split('_')[1]
        ).astype('category')
    df_metric['condition'] = df_metric['control_recording'].map(
        {True: 'control', False: 'lsd'}
        ).astype('category')

    for epoch in ['spontaneous', 'replay']:
        print("\n=================================================================")
        print(f"Mixed-effects ANOVA - {metric} - {epoch}")
        print("=================================================================")

        aov = pg.mixed_anova(
            data=df_metric.query('epoch == @epoch'),
            dv=metric,
            within='task',
            between='condition',
            subject='eid'
        )
        print(aov)
        aov.to_csv(PROJECT_ROOT / f'results/MixedANOVA_{metric}_{epoch}.csv')

        fig, ax = plt.subplots()
        # Plot each condition separately
        for condition in [True, False]:
            # Get data for this epoch and condition
            df_condition = df_pca.query('control_recording == @condition')
            xx = df_condition[f'task00_{epoch}_{metric}']
            yy = df_condition[f'task01_{epoch}_{metric}']
            # Pick colors and markers
            color = CONTROLCOLOR if condition else LSDCOLOR
            ax.scatter(xx, yy, s=10, ec=color, fc='none')
        # Plot unity line over full range of data
        xx = df_pca[f'task00_{epoch}_{metric}']
        yy = df_pca[f'task01_{epoch}_{metric}']
        ax.plot(
            [min(xx.min(), yy.min()), max(xx.max(), yy.max())],
            [min(xx.min(), yy.min()), max(xx.max(), yy.max())],
            color='gray', ls='--'
        )
        # Format axes
        ax.set_title(f'{epoch.title()} {POPMETRICLABELS[metric]}')
        # ~plots.auto_tick(ax.xaxis, nticks=3, locator_class=None)
        # ~plots.auto_tick(ax.yaxis, nticks=3, locator_class=None)
        ax.set_xticks(POPMETRICTICKS[metric])
        ax.set_yticks(POPMETRICTICKS[metric])
        ax.set_xlabel(f'Pre')
        ax.set_ylabel(f'Post')
        plots.clip_axes_to_ticks(ax=ax)
        plots.set_plotsize(w=2)
        fig.savefig(
            PROJECT_ROOT / f'figures/{epoch}_{metric}.svg'
            )


def _pc_rotation(session, contrast):
    n = max(session[f'{contrast[0]}_n80'], session[f'{contrast[1]}_n80'])
    D = cosine_similarity(
        session[f'{contrast[0]}_loadings'][:n],
        session[f'{contrast[1]}_loadings'][:n]
        )
    return D

# Define number of bins for the histogram
num_bins = 12
bins = np.linspace(0, np.pi, num_bins + 1)
# Compute bin centers for plotting
bin_centers = (bins[:-1] + bins[1:]) / 2.0
width = np.pi / num_bins

for epoch in ['spontaneous', 'replay']:
    df_pca[f'{epoch}_PCD'] = df_pca.apply(
    lambda x:
        _pc_rotation(x, [f'task00_{epoch}', f'task01_{epoch}']),
    axis='columns'
    )
    df_pca[f'{epoch}_PC1sim'] = df_pca[f'{epoch}_PCD'].apply(
        lambda x: x[0, 0]
        )
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for control in [True, False]:

        df_condition = df_pca.query('control_recording == @control')
        angles = df_condition[f'{epoch}_PCD'].apply(
            lambda x: np.arccos(x[0, 0])
            )
        weights = np.ones_like(angles) / len(angles)
        counts, _ = np.histogram(angles, bins=bins, weights=weights)
        color = CONTROLCOLOR if control else LSDCOLOR
        ax.bar(
            bin_centers,
            counts,
            width=width,
            align='center',
            fc=color,
            ec='black',
            alpha=0.5
            )
        ax.scatter(angles.mean(), 0.75, s=25, color=color)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xlabel('PC1 Rotation')
    ax.tick_params(axis='x', pad=-3)
    ax.set_title(epoch.title())
    plots.set_plotsize(w=2)
    fig.savefig(
        PROJECT_ROOT / f'figures/{epoch}_PC1rotation.svg'
        )

    # ~t, p = ttest_ind(
        # ~df_pca.query('control_recording == True')[f'{epoch}_pc1similarity'],
        # ~df_pca.query('control_recording == False')[f'{epoch}_pc1similarity']
        # ~)
    # ~print(f"Independent samples t-test (t, p): {t:.2e}, {p:.2e}")

metric_cols = [
    col for col in df_pca.columns if 'PC1sim' in col
    ]
df_metric = df_pca.melt(
    id_vars=['subject', 'eid', 'control_recording', 'trajectory_label'],
    value_vars=metric_cols,
    var_name='epoch',
    value_name='PC1sim'
    )
df_metric['epoch'] = df_metric['epoch'].apply(
    lambda x: x.split('_')[0]
    ).astype('category')
df_metric['condition'] = df_metric['control_recording'].map(
    {True: 'control', False: 'lsd'}
    ).astype('category')
# ~df_metric['trajectory'] = df_metric['trajectory_label'].astype('category')

lmm_formulas = [
    f'PC1sim ~ condition * epoch + (1 | subject)',
    ]
model = Lmer(
    lmm_formulas[0],
    data=df_metric,
    )
result = model.fit()
print(model.warnings)
if not model.warnings:
    model_converged = True
    print(result)

# ~ols_formula = f'pc1similarity ~ condition * epoch'
# ~model = smf.ols(
    # ~ols_formula,
    # ~data=df_metric
    # ~)
# ~result = model.fit()
# ~print(result.summary())

# ~aov = pg.mixed_anova(
    # ~data=df_metric.query('epoch == @epoch'),
    # ~dv='PC1sim',
    # ~within='epoch',
    # ~between='condition',
    # ~subject='eid'
# ~)
# ~print(aov)

for epoch in ['spontaneous', 'replay']:
    print("\n====")
    print(f"One-way ANOVA - PC1 similarity - {epoch}")
    print("====")
    # ~ols_formula = f'PC1sim ~ condition'
    # ~model = smf.ols(
        # ~ols_formula,
        # ~data=df_metric.query('epoch == @epoch')
        # ~)
    # ~result = model.fit()
    # ~anova_table = anova_lm(result, typ=2)  # Type II SS
    # ~print(anova_table)
    print(ttest_ind(
            df_metric.query('epoch == @epoch and condition == "lsd"')['PC1sim'],
            df_metric.query('epoch == @epoch and condition == "control"')['PC1sim']
            )
        )


def abs_max_per_row(X):
    return np.max(np.abs(X), axis=1)

def _stats(df, y):
    df_metric = df.melt(
        id_vars=['subject', 'eid', 'control_recording'],
        value_vars=y,
        var_name='epoch',
        value_name='PCsim'
        )
    df_metric['epoch'] = df_metric['epoch'].apply(
        lambda x: x.split('_')[0]
        ).astype('category')
    df_metric['condition'] = df_metric['control_recording'].map(
        {True: 'control', False: 'lsd'}
        ).astype('category')
    ols_formula = f'PCsim ~ condition'
    model = smf.ols(
        ols_formula,
        data=df_metric
        )
    result = model.fit()
    anova_table = anova_lm(result, typ=2)  # Type II SS
    print(anova_table)
    return anova_table.loc['condition', 'PR(>F)']


n_components = np.arange(2, 5)
alpha = 0.05
for epoch in ['spontaneous', 'replay']:

    # For each session, find the PC similarity considering n components
    df_pca = df_pca.assign(**{
        f'{epoch}_PCsim{n}': df_pca[f'{epoch}_PCD'].apply(
            lambda X: abs_max_per_row(X[:n, :n]).mean()
            )
        for n in n_components
    })
    df_pca = df_pca.assign(**{
        f'{epoch}_PCsim{n}_p': _stats(df_pca, f'{epoch}_PCsim{n}')
        for n in n_components
        })

    # ~for n in n_components:
        # ~df_pca = df_pca.assign(
            # ~**{f'{epoch}_PCsim{n}_p': None for n in n_components}
            # ~)
        # ~df_metric = df_pca.melt(
            # ~id_vars=['subject', 'eid', 'control_recording', 'trajectory_label'],
            # ~value_vars=f'{epoch}_PCsim{n}',
            # ~var_name='epoch',
            # ~value_name=f'PCsim'
            # ~)
        # ~df_metric['epoch'] = df_metric['epoch'].apply(
            # ~lambda x: x.split('_')[0]
            # ~).astype('category')
        # ~df_metric['condition'] = df_metric['control_recording'].map(
            # ~{True: 'control', False: 'lsd'}
            # ~).astype('category')
        # ~ols_formula = f'PCsim ~ condition'
        # ~model = smf.ols(
            # ~ols_formula,
            # ~data=df_metric
            # ~)
        # ~result = model.fit()
        # ~anova_table = anova_lm(result, typ=2)  # Type II SS
        # ~print(anova_table)
        # ~df_pca[f'{epoch}_PCsim{n}_p'] = anova_table.loc['condition', 'PR(>F)']

    fig, ax = plt.subplots()
    ax.set_title(epoch.title())
    pvalues = np.array(
        [df_pca[f'{epoch}_PCsim{n}_p'].iloc[0] for n in n_components]
        )
    reject, pvalues_corr, _, _ = multipletests(
        pvalues, alpha=alpha, method='fdr_bh'
        )
    for control in [True, False]:
        df_condition = df_pca.query('control_recording == @control')
        yy = np.stack(
            df_condition.apply(
                lambda x: [x[f'{epoch}_PCsim{n}'] for n in n_components],
                axis='columns'
                )
            )
        ax.plot(
            n_components,
            yy.mean(axis=0),
            color=CONTROLCOLOR if control else LSDCOLOR,
            )
        ax.fill_between(
            n_components,
            yy.mean(axis=0) + sem(yy, axis=0),
            yy.mean(axis=0) - sem(yy, axis=0),
            color=CONTROLCOLOR if control else LSDCOLOR,
            alpha=0.25
            )
        ax.scatter(
            # ~n_components[reject],
            # ~yy.mean(axis=0)[reject],
            n_components[pvalues <= 0.05],
            yy.mean(axis=0)[pvalues <= 0.05],
            color=CONTROLCOLOR if control else LSDCOLOR,
            s=5
            )
    ax.set_xticks([2, 50, 100])
    ax.set_xlabel('Number of Components')
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_ylabel('Mean Cosine Similarity')
    plots.clip_axes_to_ticks(ax=ax)
    plots.set_plotsize(w=4, h=4, ax=ax)
    fig.savefig(
        PROJECT_ROOT / f'figures/{epoch}_PCsimilarity.svg'
        )

exclude_cols = ['_D', '_loadings', '_PCD']
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df_pca.drop(
    columns=[
        col for col in df_pca.columns
        if any([x in col for x in exclude_cols])
        ]
    ).to_parquet(PROJECT_ROOT / f'data/popdim_{timestamp}.pqt')


#### Per Region ################################################################

def popdim(df_spikes, epochs, group_by=['eid', 'coarse_region']):
    # Useful metadata for analysis downstream
    columns_to_keep = ['subject', 'eid', 'session_n', 'control_recording', 'start_time', 'coarse_region']
    pca_results = []
    for idx, group_spikes in df_spikes.groupby(group_by):
        # print(f"{idx}\nN neurons: {len(group_spikes)}\n")
        if len(group_spikes) < 25:
            continue
        group_data = {col: group_spikes[col].unique()[0] for col in columns_to_keep}
        for epoch in epochs:
            X = np.column_stack(group_spikes[f'{epoch}_counts'])
            scaler = StandardScaler()
            X_rescaled = scaler.fit_transform(X)
            # transformer = PowerTransformer(method='yeo-johnson', standardize=True) # Pre-processing step to satisfy gaussianity assumption in PCA
            # X_rescaled = transformer.fit_transform(X)
            pca = PCA()
            X_pcs = pca.fit_transform(X_rescaled)
            group_data[f'{epoch}_eigenspectrum'] = pca.explained_variance_ratio_
            group_data[f'{epoch}_loadings'] = pca.components_
            group_data[f'{epoch}_pc1'] = pca.explained_variance_ratio_[0]
            group_data[f'{epoch}_n80'] = np.cumsum(pca.explained_variance_ratio_).searchsorted(0.8)
            group_data[f'{epoch}_powlaw'] = util.power_law_slope(pca.explained_variance_ratio_)
            group_data[f'{epoch}_ngsc'] = util.normalized_entropy(pca.explained_variance_ratio_)
        pca_results.append(group_data)
    df_pca = pd.DataFrame(pca_results)
    return df_pca

df_pca = popdim(df_spikes, epochs)


for epoch in ['spontaneous', 'replay']:
    df_pca[f'{epoch}_PCD'] = df_pca.apply(
    lambda x:
        _pc_rotation(x, [f'task00_{epoch}', f'task01_{epoch}']),
    axis='columns'
    )
    df_pca[f'{epoch}_PC1sim'] = df_pca[f'{epoch}_PCD'].apply(
        lambda x: x[0, 0]
        )


n_components = np.arange(2, 25)
alpha = 0.05
for epoch in ['spontaneous', 'replay']:

    # For each session, find the PC similarity considering n components
    df_pca = df_pca.assign(**{
        f'{epoch}_PCsim{n}': df_pca[f'{epoch}_PCD'].apply(
            lambda X: abs_max_per_row(X[:n, :n]).mean()
            )
        for n in n_components
    })
    df_pca = df_pca.assign(**{
        f'{epoch}_PCsim{n}_p': _stats(df_pca, f'{epoch}_PCsim{n}')
        for n in n_components
        })


exclude_cols = ['_D', '_loadings', '_PCD']
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df_pca.drop(
    columns=[
        col for col in df_pca.columns
        if any([x in col for x in exclude_cols])
        ]
    ).to_parquet(PROJECT_ROOT / f'data/popdimRegions_{timestamp}.pqt')

for metric in metrics:
    for epoch in ['spontaneous', 'replay']:
        df_pca[f'{epoch}_{metric}_diff'] = df_pca.apply(
            lambda x:
                x[f'task01_{epoch}_ngsc'] - x[f'task00_{epoch}_ngsc'],
            axis='columns'
            )
        grouped = df_pca.groupby('coarse_region')

        fig, ax = plt.subplots()
        for i, (idx, group) in enumerate(grouped):
            for condition, offset in zip([True, False], [-0.1, 0.1]):
                group_condition = group.query('control_recording == @condition')
                color = CONTROLCOLOR if condition else LSDCOLOR
                ax.scatter(
                    np.ones(len(group_condition)) * i + offset,
                    group_condition[f'{epoch}_{metric}_diff'],
                    fc='none',
                    ec=color,
                    s=10
                    )
        ax.axhline(0, ls='--', lw=1, color='gray')
        ax.set_xticks(np.arange(len(grouped)))
        ax.set_xticklabels(grouped.groups.keys())
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('$\Delta$ %s (Post-Pre)' % metric)
        ax.set_title(f'{epoch.capitalize()}')
        plots.set_plotsize(w=12, h=6)


# Find n80 for each session
var_exp = []
for _, neurons in populations:
    X = np.stack(neurons['full_counts']).T
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA()
    X_pcs = pca.fit_transform(X_norm)
    var_exp.append(pca.explained_variance_ratio_)
fig, ax = plt.subplots()
for var in var_exp:
    ax.plot(np.cumsum(var), color='gray', alpha=0.5)
n80s = [np.cumsum(v).searchsorted(0.8) for v in var_exp]
print(f"N80 (mean, sd): {np.mean(n80s)}, {np.std(n80s)}")


## Mixed ANOVA with multiple within factors in R+ezANOVA (rpy2)
# ~import rpy2.robjects as ro
# ~from rpy2.robjects import pandas2ri
# ~from rpy2.robjects.packages import importr

# ~pandas2ri.activate()

# ~# Prepare data
# ~df_r = df_metric.copy()
# ~df_r['eid'] = df_r['eid'].astype(str)  # R needs factor

# ~# Convert to R dataframe
# ~r_df = pandas2ri.py2rpy(df_r)
# ~ro.globalenv['df'] = r_df

# ~# Run mixed ANOVA in R
# ~ro.r(f'''
# ~library(ez)

# ~result <- ezANOVA(
    # ~data = df,
    # ~dv = {metric},
    # ~wid = eid,
    # ~within = .(task, epoch),
    # ~between = condition,
    # ~type = 3,
    # ~detailed = TRUE
# ~)

# ~print(result)
# ~''')

# ~# Extract results back to Python
# ~anova_table = ro.r('result$ANOVA')
# ~print(pandas2ri.rpy2py(anova_table))


# Sliding epochs defined by time relative to some t0
t0 = 'LSD_admin'  # must refer to column that exists in the dataframe
# Use default epochs definted in config
epoch_length = postLSD_epoch_length
epoch_starts = postLSD_epoch_starts
# Or define custom epochs
epoch_length = 5 * 60  # windows are 5min long
# windows every 2.5min from 10min before t0 to 30min post
epoch_starts = np.arange(-10 * 60, 30 * 60, 2.5 * 60)
# Add the sliding epochs to the dataframe (return_cols also gives us a list of all the epochs that were added
df_spikes, sliding_epochs = util.sliding_epochs(
    df_spikes, t0=t0, epochs=epoch_starts, length=epoch_length, return_cols=True
    )

# Get spike counts for epochs
dt = 1  # seconds
for epoch in sliding_epochs:
    print(f"Getting spike counts for: {epoch}")
    # Convert spike times to spike counts in bins of size dt
    df_spikes[f'{epoch}_counts'] = df_spikes.progress_apply(
        lambda neuron:
            spikes._apply_spike_counts(
                neuron, epoch=epoch, dt=dt
                ),
        axis='columns'
        )
df_pca = popdim(df_spikes, sliding_epochs)
df = df_pca

# Pick a metric
for epoch in ['spontaneous', 'replay']:
    for metric in metrics:
        # Extract metric values from dataframe
        metric_cols = ['_'.join([epoch, metric]) for epoch in sliding_epochs]
        # Use window centers as time points
        tpts = epoch_starts + epoch_length / 2
        # Decide whether to plot individual lines for each neural population
        plot_samples = False  # if false, SEM errorbars will be plotted

        # Run an independent samples t-test for each epoch
        df_lsd = df.query('control_recording == False')
        df_saline = df.query('control_recording == True')
        ttest_results = [stats.ttest_ind(df_lsd.dropna(subset=col)[col], df_saline.dropna(subset=col)[col]) for col in metric_cols]
        pvals, tstats = [res.pvalue for res in ttest_results], [res.statistic for res in ttest_results]
        pvals = stats.false_discovery_control(pvals, method='bh')

        fig, ax = plt.subplots()
        # Plot the mean for each condition
        for condition in [True, False]:
            mean = df[df['control_recording'] == condition][metric_cols].mean(axis=0)
            color = CONTROLCOLOR if condition else LSDCOLOR
            ax.plot(tpts, mean, color=color)
            if not plot_samples:
                sem = df[df['control_recording'] == condition][metric_cols].sem(axis=0)
                ax.fill_between(tpts, mean - sem, mean + sem, color=color, alpha=0.25)

        # Plot individual neural populations
        if plot_samples:
            for idx, population in df.iterrows():
                color = CONTROLCOLOR if population['control_recording'] else LSDCOLOR
                # ~linestyle = '--' if population['first_lsd_session'] else '-'  # dashed lines indicate the first LSD session for a mouse
                ax.plot(
                    tpts, population[metric_cols],
                    color=color, linewidth=1, alpha=0.25
                    )

        # Add the p-values
        y = df[metric_cols].mean(axis=0).max()
        ns = ''  # what to display if the result isn't significant (default = 'n.s.')
        for t, p in zip(tpts, pvals):
            ax.text(t, y, plots.pval2stars(p, ns=ns))

        ax.axvline(0, ls='--', color='gray')

        # Format axes
        ax.set_title(epoch.title())
        ax.set_xticks([0] + list(tpts))
        ax.tick_params('x', rotation=45)
        ax.set_xlabel(f"Time since {t0} (s)")
        ax.set_ylabel(metric)
        plots.set_plotsize(w=8, h=4)
        fig.savefig(
            PROJECT_ROOT / f'figures/{epoch}_{metric}_sliding.svg'
        )

