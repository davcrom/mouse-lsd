import numpy as np
import pandas as pd
import re
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt

from pymer4.models import Lmer
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from psyfun import io, atlas, spikes, plots
from psyfun.config import *


epoch_pairs = {
    'spontaneous': ('task00_spontaneous', 'task01_spontaneous'),
    #'replay': ('task00_replay', 'task01_replay'),
}


metrics = {
    spikes.mean_rate: {},
    #spikes.coefficient_of_variation: {}
    }

dts = np.logspace(-1, 1, 3)
binned_metrics = {
    spikes.fano_factor: {},
    spikes.lempel_ziv_complexity: {}
    }


df_spikes = io.load_session_spikes().set_index('uuid')
df_spikes['coarse_region'] = atlas.coarse_regions(df_spikes['region'])


epoch_dfs = []
for epoch_name, epochs in epoch_pairs.items():
    df_metrics = pd.DataFrame(
        df_spikes.progress_apply(
            lambda neuron: spikes._apply_modulation_index(
                series=neuron,
                epochs=epochs,
                metrics=metrics,
                n_shf=100
            ),
            axis='columns'
        ).tolist(),
        index=df_spikes.index
    )
    
    for dt in dts:
        binned_metrics_ = {metric: {'dt': dt} for metric in binned_metrics.keys()}
        df_metrics_dt_ = pd.DataFrame(
            df_spikes.progress_apply(
                lambda neuron: spikes._apply_modulation_index(
                    series=neuron,
                    epochs=epochs,
                    metrics=binned_metrics_,
                    n_shf=100
                ),
                axis='columns'
            ).tolist(),
            index=df_spikes.index
        )
        df_metrics_dt_.columns = [f"{col}_{dt}" for col in df_metrics_dt_.columns]
        df_metrics = df_metrics.join(df_metrics_dt_)

    df_metrics.columns = [
        f"{epoch_name}_{col}" for col in df_metrics.columns
    ]
    epoch_dfs.append(df_metrics)

columns_to_keep = [
    'subject', 'eid', 'start_time', 'control_recording', 'pid', 'histology',
    'depth', 'x', 'y', 'z', 'region', 'coarse_region',
    'firing_rate',
       ]
df_singleunit = df_spikes[columns_to_keep].join(epoch_dfs)


## TEMPFIX
def reorder_column_name(col):
    """Reorder column names to put numeric suffix before task identifier"""
    # Pattern to match columns ending with task/MI/MIp followed by a number
    pattern = r'^(.+)_(task\d+|MI|MIp)_([\d.]+)$'
    match = re.match(pattern, col)

    if match:
        prefix = match.group(1)  # e.g., 'spontaneous_fano_factor'
        task = match.group(2)     # e.g., 'task00', 'MI', 'MIp'
        number = match.group(3)   # e.g., '0.1', '1.0', '10.0'

        # Convert to milliseconds
        dt_ms = float(number) * 1000

        # Format as integer if it's a whole number, otherwise keep decimal
        if dt_ms.is_integer():
            dt_str = str(int(dt_ms))
        else:
            dt_str = str(dt_ms)

        return f"{prefix}_dt{dt_str}_{task}"

    return col
df_singleunit.columns = [reorder_column_name(col) for col in df_singleunit.columns]


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df_singleunit.to_parquet(PROJECT_ROOT / f'data/singleunit_{timestamp}.pqt')


#### END OF ANALYSIS LOOP ######################################################


df_singleunit = pd.read_parquet(PROJECT_ROOT / 'data/singleunit_20251016_121250.pqt')
df_singleunit['condition'] = df_singleunit['control_recording'].apply(
    lambda x: 'Saline' if x else 'LSD'
    ).astype('category')

# Plot mean rate changes across
for epoch in epoch_pairs.keys():
    fig, ax = plt.subplots()
    ax.set_title(epoch.title())
    for control in [False, True]:
        df_condition = df_singleunit.query('control_recording == @control')
        plots.paired_plot(
            df_condition,
            [f'{epoch}_mean_rate_task00', f'{epoch}_mean_rate_task01'],
            'eid',
            transform=np.log10,
            ax=ax,
            marker='o',
            color=CONTROLCOLOR if control else LSDCOLOR,
            alpha=0.5
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['$10^{%d}$' % tick for tick in [-1, 0, 1]])
    ax.set_ylabel('Firing Rate (Hz) Session Mean')
    plots.clip_axes_to_ticks(ax=ax)
    plots.set_plotsize(w=3, h=6, ax=ax)
    fig.savefig(PROJECT_ROOT / f'figures/{epoch}_meanrate.svg')


## TODO: need to find a simpler way to do this...
metric_names = (
                [metric.__name__ for metric in metrics.keys()] +
                [metric.__name__ for metric in binned_metrics.keys()]
                )
metric_names = np.unique([
    '_'.join(col.split('_')[1:-1]) for col in df_singleunit.columns
    if any(sub in col for sub in metric_names)
    ])

# Loop over metrics, generate plots, and fit LMMs
for metric in metric_names:
    for epoch in epoch_pairs.keys():
        fig, ax = plt.subplots()
        ax.set_title(EPOCHLABELS[epoch])
        for control in [False, True]:
            df_condition = df_singleunit.query('control_recording == @control')
            plots.paired_plot(
                df_condition,
                [f'{epoch}_{metric}_task00', f'{epoch}_{metric}_task01'],
                'eid',
                ax=ax,
                marker='o',
                color=CONTROLCOLOR if control else LSDCOLOR,
                alpha=0.5
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_xlim([-0.2, 1.2])
        if metric in METRICMEANTICKS.keys():
            ax.set_yticks(METRICMEANTICKS[metric])
        else:
            plots.auto_tick(ax.yaxis)
        ax.set_ylabel(METRICLABELS[metric])
        plots.clip_axes_to_ticks(ax=ax)
        plots.set_plotsize(w=2, h=4, ax=ax)
        fig.savefig(PROJECT_ROOT / f'figures/{epoch}_{metric}.svg')


lmm_formulas = [
    # ~f'{metric} ~ task * condition * epoch + (task * epoch * condition | subject)',
    f'~ task * condition * epoch + (task * epoch | subject)',
    f'~ task * condition * epoch + (task | subject)',
    f'~ task * condition * epoch + (1 | subject)',
    ]
lmm_formulas = [
    f'~ task * condition * epoch + (1 | subject)',
    ]
ols_formula = f'~ task * condition * epoch'
aggregate_ols = True  # perform modelling on session medians, very conservative

for metric in metric_names:
    print("\n=================================================================")
    print(metric)
    print("=================================================================")
    metric_cols = [
        col for col in df_singleunit.columns if
        any(col.endswith(f'_{task}') for task in ['task00', 'task01']) and
        metric in col
        ]
    df_metric = df_singleunit.reset_index().melt(
        id_vars=['subject', 'eid', 'control_recording', 'uuid'],
        value_vars=metric_cols,
        var_name='epoch',
        value_name=metric
        )
    df_metric['task'] = df_metric['epoch'].apply(
        lambda x: x.split('_')[-1]
        ).astype('category')
    df_metric['epoch'] = df_metric['epoch'].apply(
        lambda x: x.split('_')[0]
        ).astype('category')
    df_metric['condition'] = df_metric['control_recording'].map(
        {True: 'control', False: 'lsd'}
        ).astype('category')

    # Consider log-transforming mean_rate
    # ~df_metric[metric] = df_metric[metric].apply(lambda x: np.log10(x + 1))

    model_converged = False
    i = 0
    while not model_converged:
        if i < len(lmm_formulas):
            print("\n====")
            print("LMM: ", lmm_formulas[i])
            print("====")
            model = Lmer(
                f'{metric} ' + lmm_formulas[i],
                data=df_metric,
                # ~family=Gamma(link='log')
                )
            result = model.fit()
            print(model.warnings)
            if not model.warnings:
                model_converged = True
                print(result)
                df_result = result.copy()
                df_result['formula'] = lmm_formulas[i]
                df_result.to_csv(PROJECT_ROOT / f'results/LMM_{metric}.csv')
        else:
            print("\n====")
            print("OLS: ", ols_formula)
            print("====")
            df_sessions = df_metric.groupby(
                ['eid', 'task', 'condition', 'epoch']
                )[metric].median().reset_index()
            model = smf.ols(
                f'{metric} ' + ols_formula,
                data=df_sessions if aggregate_ols else df_metric
                )
            result = model.fit()
            print(result.summary())
            df_result = smfresult2df(result)
            df_result['formula'] = ols_formula
            df_result.to_csv(PROJECT_ROOT / f'results/OLS_{metric}.csv')
            # ~anova_table = anova_lm(result, typ=2)  # Type II SS
            # ~print(anova_table)
            break
        i += 1
    print("\n\n")

## Debugging model fits
# ~# Get residuals (if pymer4 provides them)
# ~residuals = model.residuals
# ~# QQ-plot
# ~from scipy import stats
# ~fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ~# Histogram
# ~ax1.hist(residuals, bins=50, edgecolor='black')
# ~ax1.set_xlabel('Residuals')
# ~ax1.set_ylabel('Frequency')
# ~ax1.set_title('Residual Distribution')
# ~# QQ-plot
# ~stats.probplot(residuals, dist="norm", plot=ax2)
# ~ax2.set_title('Q-Q Plot')


## Closer look at a given metric
# ~metric = 'fano_factor_dt10.0'
# ~lmm_formula = f'~ task * condition + (1 | subject)'  # significiant interaction
# ~lmm_formula = f'~ task * condition + (task | subject)'  # marginal
# ~for epoch in ['spontaneous', 'replay']:
    # ~print("\n====")
    # ~print(f"{metric} - {epoch}")
    # ~print("LMM: ", lmm_formula)
    # ~print("====")

    # ~# Prepare the dataframe (convert to long format)
    # ~metric_cols = [
        # ~col for col in df_singleunit.columns if
        # ~any(col.endswith(f'_{task}') for task in ['task00', 'task01']) and
        # ~metric in col
        # ~]
    # ~df_metric = df_singleunit.reset_index().melt(
        # ~id_vars=['subject', 'eid', 'control_recording', 'uuid'],
        # ~value_vars=metric_cols,
        # ~var_name='epoch',
        # ~value_name=metric
        # ~)
    # ~df_metric['task'] = df_metric['epoch'].apply(
        # ~lambda x: x.split('_')[-1]
        # ~).astype('category')
    # ~df_metric['epoch'] = df_metric['epoch'].apply(
        # ~lambda x: x.split('_')[0]
        # ~).astype('category')
    # ~df_metric['condition'] = df_metric['control_recording'].map(
        # ~{True: 'control', False: 'lsd'}
        # ~).astype('category')

    # ~# Fit LMM for this epoch
    # ~model = Lmer(
        # ~f'{metric} ' + lmm_formula,
        # ~data=df_metric.query('epoch == @epoch'),
        # ~)
    # ~result = model.fit()
    # ~print(model.warnings)
    # ~if not model.warnings:
        # ~model_converged = True
        # ~print(result)
        # ~df_result = result.copy()
        # ~df_result['formula'] = lmm_formula
        # ~df_result.to_csv(PROJECT_ROOT / f'results/LMM_{metric}_{epoch}.csv')


## TEMPFIX
def remove_dot_column_name(col):
    """Reorder column names to put numeric suffix before task identifier"""
    # Pattern to match columns ending with task/MI/MIp followed by a number
    pattern = r'^(.+)_dt([\d.]+)_(task\d+|MI|MIp)$'
    match = re.match(pattern, col)

    if match:
        prefix = match.group(1)  # e.g., 'spontaneous_fano_factor'
        number = match.group(2)     # e.g., 'task00', 'MI', 'MIp'
        task = match.group(3)   # e.g., '0.1', '1.0', '10.0'

        # Convert to milliseconds
        dt_ms = float(number) * 1000

        # Format as integer if it's a whole number, otherwise keep decimal
        if dt_ms.is_integer():
            dt_str = str(int(dt_ms))
        else:
            dt_str = str(dt_ms)

        return f"{prefix}_dt{dt_str}_{task}"

    return col
df_singleunit.columns = [remove_dot_column_name(col) for col in df_singleunit.columns]

def fix_metric_name(col):
    """Reorder column names to put numeric suffix before task identifier"""
    # Pattern to match columns ending with task/MI/MIp followed by a number
    pattern = r'^(.+)_dt([\d.]+)$'
    match = re.match(pattern, col)

    if match:
        prefix = match.group(1)  # e.g., 'spontaneous_fano_factor'
        number = match.group(2)     # e.g., '0.1', '1.0', '10.0'

        # Convert to milliseconds
        dt_ms = float(number) * 1000

        # Format as integer if it's a whole number, otherwise keep decimal
        if dt_ms.is_integer():
            dt_str = str(int(dt_ms))
        else:
            dt_str = str(dt_ms)

        return f"{prefix}_dt{dt_str}"

    return col

#### BinomialGLMM for proportion significant per condition #####################

alpha = 0.05
df_singleunit['condition'] = df_singleunit['control_recording'].astype('category')
for metric in [fix_metric_name(col) for col in metric_names]:
    for epoch in epoch_pairs.keys():
        print("\n=================================================================")
        print(f"{metric} - {epoch} - Bonferroni-corrected alpha={alpha/len(metric_names)}")
        print("=================================================================")
        for control in [False, True]:
            print("\n====")
            print("Saline" if control else "LSD")
            print("====")
            df_condition = df_singleunit.query('control_recording == @control')
            queries = {
                'sig.': f'{epoch}_{metric}_MIp <= {alpha/2} or {epoch}_{metric}_MIp >= {1 - alpha/2}',
                'n.s.': f'{epoch}_{metric}_MIp > {alpha/2} and {epoch}_{metric}_MIp < {1 - alpha/2}'
            }

            fig, ax = plt.subplots()
            title = 'Saline' if control else 'LSD'
            ax.set_title(f'{title} - {epoch.title()}')
            color = CONTROLCOLOR if control else LSDCOLOR
            for label, query, in queries.items():
                df_query = df_condition.query(query)
                propsig = len(df_query) / len(df_condition)
                print(
                    f"{label}: {propsig:.2f} ",
                    f"({len(df_query)} / {len(df_condition)})"
                    )
                ax.hist(
                    df_query[f'{epoch}_{metric}_MI'],
                    bins=30,
                    weights=np.ones(len(df_query)) / len(df_condition),
                    histtype='step',
                    linewidth=2,
                    color=color if label == 'sig.' else 'gray',
                    label=label + f' ({(propsig * 100):.0f}%)'
                    )
            ax.set_xticks([-1, 0, 1])
            ax.set_xlabel(f'Modulation Index\n{metric}')
            plots.auto_tick(ax.yaxis)
            ax.set_ylabel('Proportion')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
            plots.clip_axes_to_ticks(ax=ax)
            plots.set_plotsize(w=6, h=3, ax=ax)
            fig.savefig(
                PROJECT_ROOT / f'figures/{epoch}_{metric}_MIdist_{title}.svg'
                )

        df_singleunit[f'{epoch}_{metric}_MIsig'] = df_singleunit[f'{epoch}_{metric}_MIp'].apply(
            lambda x:
                (x <= alpha/2) or (x >= 1 - alpha/2)
            ).astype(int)

        # Logistic mixed model
        model = Lmer(
            f'{epoch}_{metric}_MIsig ~ condition + (1 | subject)',
            data=df_singleunit.dropna(subset=f'{epoch}_{metric}_MIp'),
            family='binomial'  # For binary outcome
        )
        result = model.fit()
        if not model.warnings:
            print(result[['Estimate', '2.5_ci', '97.5_ci', 'P-val', 'Sig']])
            df_result = result.copy()
            df_result['formula'] = lmm_formula
            df_result.to_csv(PROJECT_ROOT / f'results/GLMM_{metric}_{epoch}.csv')


#### MI Distribution Diff ######################################################
from scipy.stats import ks_2samp, gaussian_kde
# ~df_singleunit['condition'] = df_singleunit['control_recording'].apply(
    # ~lambda x: 'Saline' if x else 'LSD'
    # ~).astype('category')
xvals = np.linspace(-1, 1, 1000)
alpha = 0.5
for metric in metric_names:
    for epoch in epoch_pairs.keys():
        data_col = f'{epoch}_{metric}_MI'
        query = f'({epoch}_{metric}_MIp <= {alpha/2}) or ({epoch}_{metric}_MIp >= {1 - alpha/2})'
        # ~df_epoch = df_singleunit.query(query)
        df_epoch = df_singleunit.copy()
        pdfs = df_epoch.dropna(subset=data_col).groupby('condition').apply(
            lambda x:
                gaussian_kde(x[data_col])(xvals)
                if len(x) > 10 else np.full(len(xvals), np.nan),
            include_groups=False
            )
        ks_stat, ks_pval = ks_2samp(*[
            df_epoch[df_epoch['condition'] == c][data_col].dropna()
            for c in df_epoch['condition'].unique()
            ])
        print(ks_pval)
        pdf_diff = np.diff(np.flip(np.stack(pdfs)), axis=0).squeeze()
        pdf_control = pdfs.loc['Saline']   # Assuming True = control
        pdf_lsd = pdfs.loc['LSD']       # Assuming False = LSD
        # Explicit subtraction (LSD - Control)
        pdf_diff = pdf_lsd - pdf_control
        fig, ax = plt.subplots()
        title = 'Saline' if control else 'LSD'
        ax.set_title(EPOCHLABELS[epoch])
        color = CONTROLCOLOR if control else LSDCOLOR
        ax.plot(xvals, pdf_diff, color='black')
        ax.fill_between(
            xvals, pdf_diff, where=pdf_diff >= 0,
            color=LSDCOLOR, alpha=0.25
            )
        ax.fill_between(
            xvals, pdf_diff, where=pdf_diff <= 0,
            color=CONTROLCOLOR, alpha=0.25
            )
        # Add significance stars
        ax.text(
            xvals[-1] + 0.05 * (xvals[-1] - xvals[0]),
            0,
            plots.pval2stars(ks_pval, ns=''),
            fontsize=LABELFONTSIZE,
            va='center',
            rotation=90
        )
        ax.errorbar(
            np.nanmean(df_singleunit.query('condition == "LSD"')[f'{epoch}_{metric}_MI']),
            0.45,
            xerr=df_singleunit.query('condition == "LSD"').dropna(subset=f'{epoch}_{metric}_MI')[f'{epoch}_{metric}_MI'].sem() * 1.96,
            color=LSDCOLOR,
            marker='o',
            ms=5
            )
        ax.errorbar(
            np.nanmean(df_singleunit.query('condition == "Saline"')[f'{epoch}_{metric}_MI']),
            0.34,
            xerr=df_singleunit.query('condition == "Saline"').dropna(subset=f'{epoch}_{metric}_MI')[f'{epoch}_{metric}_MI'].sem() * 1.96,
            color=CONTROLCOLOR,
            marker='o',
            ms=5
            )
        ax.set_xticks([-1, 0, 1])
        ax.set_xlabel(f'Modulation Index\n{METRICLABELS[metric]}')
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_ylabel(
            '$\Delta$ Proportion\n(LSD - Saline)'
            )
        plots.clip_axes_to_ticks(ax=ax)
        plots.set_plotsize(w=3, h=2, ax=ax)
        fig.savefig(
            PROJECT_ROOT / f'figures/{epoch}_{metric}_MIdistdiff.svg'
            )

for metric in ['mean_rate', 'fano_factor_dt10000']:
    for epoch in ['spontaneous', 'replay']:
        print(epoch, metric)
        print(ttest_ind(
                df_singleunit.query('condition == "LSD"').dropna(subset=f'{epoch}_{metric}_MI')[f'{epoch}_{metric}_MI'],
                df_singleunit.query('condition == "Saline"').dropna(subset=f'{epoch}_{metric}_MI')[f'{epoch}_{metric}_MI']
                )
            )



#### SOME CODE FROM CLAUDE ####################################################

from scipy.stats import chi2_contingency

alpha = 0.05
# Preparing for chi-squared test
df_singleunit['condition'] = df_singleunit['control_recording'].astype('category')

results_list = []

for metric in [fix_metric_name(col) for col in metric_names]:
    for epoch in epoch_pairs.keys():
        print("\n=================================================================")
        print(f"{metric} - {epoch} - Bonferroni-corrected alpha={alpha/len(metric_names)}")
        print("=================================================================")
        for control in [False, True]:
            print("\n====")
            print("Saline" if control else "LSD")
            print("====")
            df_condition = df_singleunit.query('control_recording == @control')
            queries = {
                'sig.': f'{epoch}_{metric}_MIp <= {alpha/2} or {epoch}_{metric}_MIp >= {1 - alpha/2}',
                'n.s.': f'{epoch}_{metric}_MIp > {alpha/2} and {epoch}_{metric}_MIp < {1 - alpha/2}'
            }

            fig, ax = plt.subplots()
            title = 'Saline' if control else 'LSD'
            ax.set_title(f'{title} - {epoch.title()}')
            color = CONTROLCOLOR if control else LSDCOLOR
            for label, query, in queries.items():
                df_query = df_condition.query(query)
                propsig = len(df_query) / len(df_condition)
                print(
                    f"{label}: {propsig:.2f} ",
                    f"({len(df_query)} / {len(df_condition)})"
                    )
                ax.hist(
                    df_query[f'{epoch}_{metric}_MI'],
                    bins=30,
                    weights=np.ones(len(df_query)) / len(df_condition),
                    histtype='step',
                    linewidth=2,
                    color=color if label == 'sig.' else 'gray',
                    label=label + f' ({(propsig * 100):.0f}%)'
                    )
            ax.set_xticks([-1, 0, 1])
            ax.set_xlabel(f'Modulation Index\n{metric}')
            plots.auto_tick(ax.yaxis)
            ax.set_ylabel('Proportion')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
            plots.clip_axes_to_ticks(ax=ax)
            plots.set_plotsize(w=6, h=3, ax=ax)
            fig.savefig(
                PROJECT_ROOT / f'figures/{epoch}_{metric}_MIdist_{title}.svg'
                )

        # Create separate significance columns for positive and negative
        df_singleunit[f'{epoch}_{metric}_MIsig_pos'] = (
            (df_singleunit[f'{epoch}_{metric}_MIp'] <= alpha) &  # Significant (two-tailed)
            (df_singleunit[f'{epoch}_{metric}_MI'] > 0)          # Positive MI
        ).astype(int)

        df_singleunit[f'{epoch}_{metric}_MIsig_neg'] = (
            (df_singleunit[f'{epoch}_{metric}_MIp'] <= alpha) &  # Significant (two-tailed)
            (df_singleunit[f'{epoch}_{metric}_MI'] < 0)          # Negative MI
        ).astype(int)

        df_singleunit[f'{epoch}_{metric}_MIsig'] = (
            df_singleunit[f'{epoch}_{metric}_MIsig_pos'] |
            df_singleunit[f'{epoch}_{metric}_MIsig_neg']
        ).astype(int)

        # Chi-squared test for positive modulation
        print("\n--- Positive Modulation ---")
        contingency_pos = pd.crosstab(
            df_singleunit['control_recording'],
            df_singleunit[f'{epoch}_{metric}_MIsig_pos']
        )
        print(contingency_pos)
        chi2_pos, p_pos, dof_pos, expected_pos = chi2_contingency(contingency_pos)
        print(f"Chi-squared: χ²={chi2_pos:.3f}, p={p_pos:.4f}, df={dof_pos}")

        # Chi-squared test for negative modulation
        print("\n--- Negative Modulation ---")
        contingency_neg = pd.crosstab(
            df_singleunit['control_recording'],
            df_singleunit[f'{epoch}_{metric}_MIsig_neg']
        )
        print(contingency_neg)
        chi2_neg, p_neg, dof_neg, expected_neg = chi2_contingency(contingency_neg)
        print(f"Chi-squared: χ²={chi2_neg:.3f}, p={p_neg:.4f}, df={dof_neg}")

        # Store results
        results_list.append({
            'metric': metric,
            'epoch': epoch,
            'modulation': 'positive',
            'chi2': chi2_pos,
            'p_value': p_pos,
            'dof': dof_pos
        })
        results_list.append({
            'metric': metric,
            'epoch': epoch,
            'modulation': 'negative',
            'chi2': chi2_neg,
            'p_value': p_neg,
            'dof': dof_neg
        })

# Save all results
df_results = pd.DataFrame(results_list)
df_results.to_csv(PROJECT_ROOT / 'results/chi2_modulation_tests.csv', index=False)
print("\n" + "="*60)
print("Summary of significant results (p < 0.05):")
print("="*60)
print(df_results[df_results['p_value'] < 0.05])



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define your metrics and epochs
epochs = list(epoch_pairs.keys())  # e.g., ['spontaneous', 'replay']
metrics = [fix_metric_name(col) for col in metric_names]

# Initialize storage for proportions
prop_control = np.zeros((len(metrics), len(epochs) * 2))  # *2 for pos and neg columns
prop_lsd = np.zeros((len(metrics), len(epochs) * 2))

# Create column labels
col_labels = []
for epoch in epochs:
    col_labels.extend([f'{epoch.title()} (+)', f'{epoch.title()} (-)'])

# Calculate proportions for each metric and epoch
for i, metric in enumerate(metrics):
    for j, epoch in enumerate(epochs):
        mi_col = f'{epoch}_{metric}_MI'
        sig_col = f'{epoch}_{metric}_MIsig'

        # Control condition
        control_data = df_singleunit[df_singleunit['control_recording'] == True]
        sig_control = control_data[sig_col] == True

        pos_control = ((control_data[mi_col] > 0) & sig_control).sum() / len(control_data)
        neg_control = ((control_data[mi_col] < 0) & sig_control).sum() / len(control_data)
        prop_control[i, j*2] = pos_control
        prop_control[i, j*2 + 1] = neg_control

        # LSD condition
        lsd_data = df_singleunit[df_singleunit['control_recording'] == False]
        sig_lsd = lsd_data[sig_col] == True

        pos_lsd = ((lsd_data[mi_col] > 0) & sig_lsd).sum() / len(lsd_data)
        neg_lsd = ((lsd_data[mi_col] < 0) & sig_lsd).sum() / len(lsd_data)
        prop_lsd[i, j*2] = pos_lsd
        prop_lsd[i, j*2 + 1] = neg_lsd

# Calculate difference
diff_matrix = prop_lsd - prop_control

# Determine common color scale
vmax = max(prop_control.max(), prop_lsd.max())
vmax_diff = np.abs(diff_matrix).max()

# Plot control
fig, ax = plt.subplots()
sns.heatmap(prop_control, ax=ax,
            cmap=mpl.colors.LinearSegmentedColormap.from_list("control_cmap", ['white', CONTROLCOLOR]),
            vmin=0, vmax=vmax,
            xticklabels=col_labels,
            yticklabels=metrics,
            cbar_kws={'label': 'Proportion'})
ax.set_title('Saline')
# ~axes[0].set_xlabel('')
# ~axes[0].set_ylabel('Metric')
plots.set_plotsize(w=6, h=8, ax=ax)
fig.savefig(PROJECT_ROOT / 'figures/MIheatmap_saline.svg')

# Plot LSD
fig, ax = plt.subplots()
sns.heatmap(prop_lsd, ax=ax,
            cmap=mpl.colors.LinearSegmentedColormap.from_list("LSD_cmap", ['white', LSDCOLOR]),
            vmin=0, vmax=vmax,
            xticklabels=col_labels,
            yticklabels=metrics,
            cbar_kws={'label': 'Proportion'})
ax.set_title('LSD')
# ~axes[1].set_xlabel('')
# ~axes[1].set_ylabel('')
plots.set_plotsize(w=6, h=8, ax=ax)
fig.savefig(PROJECT_ROOT / 'figures/MIheatmap_LSD.svg')

# Plot difference
# Create a diverging colormap with control color (negative), white (center), LSD color (positive)
diverging_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "control_white_lsd",
    [CONTROLCOLOR, 'white', LSDCOLOR]
)
fig, ax = plt.subplots()
sns.heatmap(diff_matrix, ax=ax, cmap=diverging_cmap, center=0,
            vmin=-vmax_diff, vmax=vmax_diff,
            xticklabels=col_labels,
            yticklabels=metrics,
            cbar_kws={'label': 'Difference (LSD - Control)'})
ax.set_title('LSD - Control')
plots.set_plotsize(w=6, h=8, ax=ax)
fig.savefig(PROJECT_ROOT / 'figures/MIheatmap_diff.svg')




## TEMPFIX
def add_dot_column_name(col):
    """Convert millisecond dt values back to seconds with decimal point"""
    # Pattern to match columns with _dt[number]_ in the middle
    pattern = r'^(.+)_dt([\d.]+)_(task\d+|MI|MIp|MIsig)$'
    match = re.match(pattern, col)

    if match:
        prefix = match.group(1)  # e.g., 'spontaneous_fano_factor'
        number = match.group(2)  # e.g., '100', '1000', '10000'
        task = match.group(3)    # e.g., 'task00', 'MI', 'MIp'

        # Convert from milliseconds back to seconds
        dt_sec = float(number) / 1000

        # Format with decimal point
        dt_str = str(dt_sec)

        return f"{prefix}_dt{dt_str}_{task}"

    return col

df_singleunit.columns = [add_dot_column_name(col) for col in df_singleunit.columns]


exclude_regions = ['None', 'Fiber tract']
for metric in metric_names:
    for epoch in epoch_pairs.keys():
        fig, ax = plots.plot_proportion_by_group(
            df_singleunit.query('coarse_region not in @exclude_regions'),
            f'{epoch}_{metric}_MIsig',
            'coarse_region',
            'control_recording',
            sort_by=False,
            colors=[LSDCOLOR, CONTROLCOLOR]
            )
        ax.set_title(f'{epoch.title()} - {metric} MI')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('Proportion Significant')
        plots.clip_axes_to_ticks(ax=ax, ext={'bottom':[-0.5, 0.5]})
        plots.set_plotsize(w=12, h=6, ax=ax)
        fig.savefig(
            PROJECT_ROOT / f'figures/{epoch}_{metric}_MIp_regions.svg'
            )


#### Mean MI per Region ########################################################

def _apply_OLS(group_data, col, min_n=10):
    group_data.reset_index(drop=True, inplace=True)
    n_per_condition = (
        sum(group_data['control_recording'] == True),
        sum(group_data['control_recording'] == False)
        )
    if min(n_per_condition) < min_n:
        print(f"{group_data.name}: skipping, too few neurons per condition!")
        return np.nan
    # ~model = Lmer(f'{col} ~ condition + (1 | subject)', data=group_data,)
    model = smf.ols(f'{col}  ~ condition', group_data)
    result = model.fit()
    # ~return result['P-val'].iloc[1]
    return result.pvalues['condition[T.Saline]']

exclude_regions = ['None', 'Fiber tract']
# ~df_singleunit.columns = [remove_dot_column_name(col) for col in df_singleunit.columns]
# ~for metric in [fix_metric_name(col) for col in metric_names]:
for metric in metric_names:
    for epoch in epoch_pairs.keys():
        # Run the stats
        p_vals = df_singleunit.groupby('coarse_region').apply(
            lambda x: _apply_OLS(x, f'{epoch}_{metric}_MI'),
            include_groups=False
            )
        print("\n=============================================================")
        print(f"OLS: {metric} - {epoch}")
        print("=============================================================")
        print(p_vals[[r not in exclude_regions for r in p_vals.index]])
        p_vals_filtered = p_vals[[r not in exclude_regions for r in p_vals.index]]
        # p_vals.to_csv(PROJECT_ROOT / f'results/{epoch}_{metric}_meanMI_regions.csv')

        # Plot with faded bars for NaN p-values
        df_plot = df_singleunit.query('coarse_region not in @exclude_regions').copy()
        na_regions = p_vals[p_vals.isna()].index
        df_plot = df_plot.query('coarse_region not in @na_regions')

        def _compare_conditions(group_data):
            lsd_data = group_data.query('condition == "LSD"')[f'{epoch}_{metric}_MI']
            saline_data = group_data.query('condition == "Saline"')[f'{epoch}_{metric}_MI']
            return np.abs(lsd_data.mean() - saline_data.mean()) * np.sign(lsd_data.mean())

        fig, ax = plots.plot_mean_by_group(
            df_plot,
            f'{epoch}_{metric}_MI',
            'coarse_region',
            'condition',
            sort_by='LSD',
            # ~agg_func=_compare_conditions,
            colors=[LSDCOLOR, CONTROLCOLOR],
            orientation='horizontal',
            ascending=True
            )

        # Add alpha column based on whether p-value is NaN
        region_alpha = {
            region: 0.3
            if np.isnan(p_vals_filtered.get(region, np.nan)) else 1.0
            for region in df_plot['coarse_region'].unique()
            }

        # Fade bars for regions with NaN p-values
        for i, region in enumerate(ax.get_yticklabels()):
            region_name = region.get_text()
            if region_name in region_alpha and region_alpha[region_name] < 1.0:
                # Find and fade all patches (bars) at this y-position
                for patch in ax.patches:
                    # Check if patch is at this y-level (within tolerance)
                    if abs(patch.get_y() + patch.get_height()/2 - i) < 0.5:
                        patch.set_alpha(0.3)

        ax.set_title(f'{epoch.title()} - {metric}')
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xlabel('Mean MI')
        plots.clip_axes_to_ticks(ax=ax, ext={'left':[-0.5, 0.5]})
        plots.set_plotsize(w=3, h=6, ax=ax)
        fig.savefig(
            PROJECT_ROOT / f'figures/{epoch}_{metric}_MI_regions.svg'
            )

include_regions = ['Prefrontal Ctx', 'Supp. Motor Ctx  
for metric in metric_names:
    for epoch in epoch_pairs.keys():
        # Plot with faded bars for NaN p-values
        df_plot = df_singleunit.query('coarse_region not in @exclude_regions').copy()
        df_plot = df_plot.query('coarse_region not in @na_regions')

        def _compare_conditions(group_data):
            lsd_data = group_data.query('condition == "LSD"')[f'{epoch}_{metric}_MI']
            saline_data = group_data.query('condition == "Saline"')[f'{epoch}_{metric}_MI']
            return np.abs(lsd_data.mean() - saline_data.mean()) * np.sign(lsd_data.mean())

        fig, ax = plots.plot_mean_by_group(
            df_plot,
            f'{epoch}_{metric}_MI',
            'coarse_region',
            'condition',
            sort_by='LSD',
            # ~agg_func=_compare_conditions,
            colors=[LSDCOLOR, CONTROLCOLOR],
            orientation='horizontal',
            ascending=True
            )

        # Add alpha column based on whether p-value is NaN
        region_alpha = {
            region: 0.3
            if np.isnan(p_vals_filtered.get(region, np.nan)) else 1.0
            for region in df_plot['coarse_region'].unique()
            }

        # Fade bars for regions with NaN p-values
        for i, region in enumerate(ax.get_yticklabels()):
            region_name = region.get_text()
            if region_name in region_alpha and region_alpha[region_name] < 1.0:
                # Find and fade all patches (bars) at this y-position
                for patch in ax.patches:
                    # Check if patch is at this y-level (within tolerance)
                    if abs(patch.get_y() + patch.get_height()/2 - i) < 0.5:
                        patch.set_alpha(0.3)

        ax.set_title(f'{epoch.title()} - {metric}')
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xlabel('Mean MI')
        plots.clip_axes_to_ticks(ax=ax, ext={'left':[-0.5, 0.5]})
        plots.set_plotsize(w=3, h=6, ax=ax)


## WIP: MI dist pdf difference (LSD - control)
for metric in ['fano_factor_dt10000', 'mean_rate']:
    df_singleunit[f'{epoch}_{metric}_MIsig'] = df_singleunit[f'{epoch}_{metric}_MIp'].apply(
        lambda x:
            (x <= alpha/2) or (x >= 1 - alpha/2)
        ).astype(int)
    for epoch in epoch_pairs.keys():
        fig, ax = plots.plot_difference_by_group(
            df_singleunit.query(
                f'{epoch}_{metric}_MIsig == 1 and coarse_region not in @exclude_regions'
                # ~'coarse_region not in @exclude_regions'
                ),
            f'{epoch}_{metric}_MI',
            'coarse_region',
            'control_recording',
            f'{epoch}_{metric}_MIsig',
            sort_by=False,
            ascending=True,
            colors={0:'gray', 1:'black'},
            alphas={0:0.5, 1:1},
            fill_color=[LSDCOLOR, CONTROLCOLOR]
            )
        ax.set_title(f'{metric} - {epoch.title()}')
        ax.set_xticks([-1, 0, 1])
        ax.set_xlabel('Mean Rate MI')
        ax.yaxis.set_ticks_position('right')
        plots.clip_axes_to_ticks(ax=ax, spines=['bottom'])
        plots.set_plotsize(w=6, h=12, ax=ax)
        fig.savefig(
            PROJECT_ROOT / f'figures/{epoch}_{metric}_MI_distdiff.svg'
            )

