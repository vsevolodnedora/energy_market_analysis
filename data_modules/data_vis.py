import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# def plot_time_series_with_residuals(
#         tasks, run_label, target, ylabel, **kwargs
# ):
#     '''
#
#     Plots forecasts split into adjacent windows. Each window can show several forecasts performed with
#     different methods listed in 'tasks', where each entry is a dictionary with 'results', 'metrics' and 'forecast'.
#     Last panel shows the latest forecast (as given in 'forecast' in tasks) and last metrics from 'metrics' in tasks.
#     Bottom panels show residuals between actual target variable and forecasted.
#
#     :param tasks: list[dict] where each dict represents a model's forecasting results that consist of
#     - tasks[0]['results']:list[pd.DataFrame] list of forecasts for past forecasting winows where each dataframe has:
#     f'{target}_actual', f'{target}_fitted', f'{target}_lower', 'f'{target}_upper'
#     - tasks[0]['metrics]:list[dict] list of performance metrics for each forecasted window (RMSE,sMAPE...) where the last
#     element in the list contains the average metrics
#     tasks[0]['forecast']:pd.Dataframe -- same as dataframes in 'results' but with the current latest forecast
#     :param target: str target name
#     :param label: y-label for the top panels
#     :param kwargs: additional arguments for plotting
#     :return: None
#     '''
#
#
#
#     # Determine the maximum number of results across tasks
#     max_n_results = max(len(task['results']) for task in tasks)
#     n_cols = max_n_results + 1  # Plus one for 'forecast'
#
#     drawstyle='default'#'steps'
#
#     # Create figure and axes
#     fig, axes = plt.subplots(
#         2, n_cols, figsize=(n_cols * 5, 8),
#         gridspec_kw={'height_ratios': [3, 1],'hspace': 0.02, 'wspace': 0.01},
#         sharex='col', sharey='row'
#     )
#
#     # Define column names
#     actual_col = f'{target}_actual'
#     fitted_col = f'{target}_fitted'
#     lower_col = f'{target}_lower'
#     upper_col = f'{target}_upper'
#
#     # For each column
#     for i in range(n_cols):
#         ax_top = axes[0, i]
#         ax_bottom = axes[1, i]
#
#         # Flag to check if 'Actual' data has been plotted
#         actual_plotted = False
#
#         # For each task
#         for task in tasks:
#             name = task['name']
#             color = task['color']
#             ci_alpha=task['ci_alpha']
#             lw=task['lw']
#             # Determine if the task has data for this column
#             if i < len(task['results']):
#                 df = task['results'][i]
#                 errs = task['metrics'][i]
#             elif i == max_n_results:
#                 df = task['forecast']
#                 errs = task['metrics'][i]
#             else:
#                 continue  # Skip plotting for this task in this column
#
#             # Plot 'Actual' data once per subplot
#             if not actual_plotted and i != n_cols-1:
#                 ax_top.plot(df.index, df[actual_col], label='Actual', color='black', drawstyle=drawstyle, lw=1.5)
#                 actual_plotted = True
#
#             # Plot fitted data
#             label = name + ' '
#             if errs is not None and i != n_cols-1:
#                 label +=  fr"RMSE={errs['rmse']:.1f}" + fr" sMAPE={errs['smape']:.1f}"
#             else:
#                 label += fr"$\langle$RMSE$\rangle$={errs['rmse']:.1f}" \
#                          + fr" $\langle$sMAPE$\rangle$={errs['smape']:.1f}"
#             ax_top.plot(df.index, df[fitted_col], label=label, color=color, drawstyle=drawstyle, lw=lw)
#
#             # Plot confidence intervals
#             ax_top.fill_between(df.index, df[lower_col], df[upper_col], color=color, alpha=ci_alpha)
#
#             # Plot residuals in the bottom panel
#             residuals = (df[actual_col] - df[fitted_col]) #/ df[actual_col]
#             ax_bottom.plot(df.index, residuals, label=name, color=color, drawstyle=drawstyle, lw=lw)
#
#             # limit plots
#             ax_bottom.set_xlim(df.index.min(),df.index.max())
#             if 'ylim0' in kwargs: ax_top.set_ylim(kwargs['ylim0'][0], kwargs['ylim0'][1])
#             if 'ylim1' in kwargs: ax_bottom.set_ylim(kwargs['ylim1'][0], kwargs['ylim1'][1])
#
#         # Add a horizontal line at y=0 in residuals plot
#         ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=1)
#
#         # Set titles and labels
#         if i < max_n_results:
#             ax_top.set_title(f'Past Forecast {max_n_results-i}', fontsize=14, weight='bold')
#         else:
#             ax_top.set_title('Current Forecast')
#
#         if ylabel and i == 0:
#             ax_top.set_ylabel(ylabel)
#             ax_bottom.set_ylabel('Residual / Actual')
#
#         # legend in the empty area in residual plots
#         if i == n_cols - 1:
#             ax_bottom.legend(loc='upper left', ncol=1, fontsize=10)
#         ax_top.legend(loc='lower left', ncol=1, fontsize=10)
#
#         for ax in [ax_top, ax_bottom]:
#             ax.grid(True, linestyle='--', alpha=0.6)
#             ax.tick_params(axis='x', direction='in', bottom=True)
#             ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
#
#         # Improve x-axis formatting
#         ax_bottom.set_xlabel(f'Date (month-day for $2024$)', fontsize=12)
#         ax_bottom.xaxis.set_major_locator(mdates.DayLocator())
#         ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#         fig.autofmt_xdate(rotation=45)
#     model_names = "".join(task["name"]+'_' for task in tasks)
#     plt.savefig(f'{run_label}_{model_names}.png', bbox_inches='tight')
#     plt.show()
#

def plot_time_series_with_residuals(
        tasks, run_label, target, ylabel, **kwargs
):
    '''

    Plots forecasts split into adjacent windows. Each window can show several forecasts performed with
    different methods listed in 'tasks', where each entry is a dictionary with 'results', 'metrics' and 'forecast'.
    Last panel shows the latest forecast (as given in 'forecast' in tasks) and last metrics from 'metrics' in tasks.
    Bottom panels show residuals between actual target variable and forecasted.

    :param tasks: list[dict] where each dict represents a model's forecasting results that consist of
    - tasks[0]['results']:list[pd.DataFrame] list of forecasts for past forecasting winows where each dataframe has:
    f'{target}_actual', f'{target}_fitted', f'{target}_lower', 'f'{target}_upper'
    - tasks[0]['metrics]:list[dict] list of performance metrics for each forecasted window (RMSE,sMAPE...) where the last
    element in the list contains the average metrics
    tasks[0]['forecast']:pd.Dataframe -- same as dataframes in 'results' but with the current latest forecast
    :param target: str target name
    :param label: y-label for the top panels
    :param kwargs: additional arguments for plotting
    :return: None
    '''



    # Determine the maximum number of results across tasks
    max_n_results = max(len(task['results']) for task in tasks)
    n_cols = max_n_results + 1  # Plus one for 'forecast'

    drawstyle='default'#'steps'

    # Create figure and axes
    # TODO fix this. Now We are plotting several targets. Each of them has its own main panel and residual panel
    # TODO Each target should be plotted in its own panel and should have its own panel for residuals
    fig, axes = plt.subplots(
        2, n_cols, figsize=(n_cols * 5, 8),
        gridspec_kw={'height_ratios': [3, 1],'hspace': 0.02, 'wspace': 0.01},
        sharex='col', sharey='row'
    )

    # Define column names
    actual_col = f'{target}_actual'
    fitted_col = f'{target}_fitted'
    lower_col = f'{target}_lower'
    upper_col = f'{target}_upper'

    # For each column
    for i in range(n_cols):
        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        # Flag to check if 'Actual' data has been plotted
        actual_plotted = False

        # For each task
        for task in tasks:
            # TODO: Fix the plot assuming that instead of name there is names which is a list with names for each target
            # TODO Fix it also for color -> colors, ci_alpha -> ci_alphas and w -> ws
            name = task['name']
            color = task['color']
            ci_alpha=task['ci_alpha']
            lw=task['lw']

            # Determine if the task has data for this column
            # TODO fix the code assuming that task.keys() = [target_1, target_2, target_3 ...]
            if i < len(task['results']):
                df = task['results'][i]
                errs = task['metrics'][i]
            elif i == max_n_results:
                df = task['forecast']
                errs = task['metrics'][i]
            else:
                continue  # Skip plotting for this task in this column

            # Plot 'Actual' data once per subplot
            if not actual_plotted and i != n_cols-1:
                ax_top.plot(df.index, df[actual_col], label='Actual', color='black', drawstyle=drawstyle, lw=1.5)
                actual_plotted = True

            # Plot fitted data
            label = name + ' '
            if errs is not None and i != n_cols-1:
                label +=  fr"RMSE={errs['rmse']:.1f}" + fr" sMAPE={errs['smape']:.1f}"
            else:
                label += fr"$\langle$RMSE$\rangle$={errs['rmse']:.1f}" \
                         + fr" $\langle$sMAPE$\rangle$={errs['smape']:.1f}"
            ax_top.plot(df.index, df[fitted_col], label=label, color=color, drawstyle=drawstyle, lw=lw)

            # Plot confidence intervals
            if lower_col in df.columns and upper_col in df.columns:
                ax_top.fill_between(df.index, df[lower_col], df[upper_col], color=color, alpha=ci_alpha)

            # Plot residuals in the bottom panel
            residuals = (df[actual_col] - df[fitted_col]) #/ df[actual_col]
            ax_bottom.plot(df.index, residuals, label=name, color=color, drawstyle=drawstyle, lw=lw)

            # limit plots
            ax_bottom.set_xlim(df.index.min(),df.index.max())
            if 'ylim0' in kwargs: ax_top.set_ylim(kwargs['ylim0'][0], kwargs['ylim0'][1])
            if 'ylim1' in kwargs: ax_bottom.set_ylim(kwargs['ylim1'][0], kwargs['ylim1'][1])

        # Add a horizontal line at y=0 in residuals plot
        ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=1)

        # Set titles and labels
        if i < max_n_results:
            ax_top.set_title(f'Past Forecast {max_n_results-i}', fontsize=14, weight='bold')
        else:
            ax_top.set_title('Current Forecast')

        if ylabel and i == 0:
            ax_top.set_ylabel(ylabel)
            ax_bottom.set_ylabel('Residual / Actual')

        # legend in the empty area in residual plots
        if i == n_cols - 1:
            ax_bottom.legend(loc='upper left', ncol=1, fontsize=10)
        ax_top.legend(loc='lower left', ncol=1, fontsize=10)

        for ax in [ax_top, ax_bottom]:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', direction='in', bottom=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)

        # Improve x-axis formatting
        ax_bottom.set_xlabel(f'Date (month-day for $2024$)', fontsize=12)
        ax_bottom.xaxis.set_major_locator(mdates.DayLocator())
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig.autofmt_xdate(rotation=45)
    model_names = "".join(task["name"]+'_' for task in tasks)
    plt.savefig(f'{run_label}_{model_names}.png', bbox_inches='tight')
    plt.show()



def plot_time_series_with_residuals_multi(
        tasks,
        run_label,
        targets=None,
        ylabels=None,
        **kwargs
):
    """
    Plots multi-target forecasts where each target has its own main panel (top)
    and residuals panel (bottom). Each column corresponds to a past forecast
    window or the current forecast.

    Parameters
    ----------
    tasks : list[dict]
        A list of tasks, where each task is a dictionary. For each target_key in
        'targets', you have something like:
          task[target_key] = {
             'results': list[pd.DataFrame],
             'metrics': list[dict],
             'forecast': pd.DataFrame
          }
        Additionally, each task has:
            - 'names':   list of model names, one per target
            - 'colors':  list of colors, one per target
            - 'ci_alphas': list of alpha (transparency) values for forecast CIs
            - 'ws':      list of line widths
    run_label : str
        Label used in the output figure name.
    targets : list[str], optional
        List of target names/keys to be plotted. If None, we will deduce from
        the first task's keys (excluding 'names', 'colors', etc.)
    ylabels : list[str], optional
        List of y-axis labels, one for each target. If None, no specialized
        y-label is used.
    kwargs : dict
        - ylim0 : tuple (ymin, ymax) for the main (top) subplots
        - ylim1 : tuple (ymin, ymax) for the residuals (bottom) subplots
        - other kwargs as needed
    """

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # If user didn't provide explicit targets, figure them out from the first task
    if targets is None:
        # We assume that the keys that store actual forecast data are everything
        # except the "names", "colors", "ci_alphas", and "ws".
        ignore_keys = {"names", "colors", "ci_alphas", "ws"}
        all_keys = list(tasks[0].keys())
        targets = [k for k in all_keys if k not in ignore_keys]

    # If user didn't provide ylabels, build placeholders
    if ylabels is None:
        ylabels = ["" for _ in targets]

    # For each target, the maximum # of results across all tasks
    # (we will assume all tasks have the same # of windows for that target)
    max_n_results = 0
    for tkey in targets:
        # Example: for each task, len(task[tkey]['results']) is # of past windows
        max_n_results = max(
            max_n_results,
            max(len(task[tkey]['results']) for task in tasks)
        )
    n_cols = max_n_results + 1  # +1 for the "current forecast"

    # For multi-target, we create 2 rows per target
    # => total rows = 2 * len(targets)
    n_targets = len(targets)
    n_rows = 2 * n_targets

    # Create the figure and axes array
    # You can adjust the figsize or height_ratios as you wish
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.5, n_targets * 4.5),
        gridspec_kw={'height_ratios': [3, 1] * n_targets,
                     'hspace': 0.02, 'wspace': 0.01},
        sharex='col',  # share x among columns
        sharey='row',
        # sharey='row'  # Not always ideal if different targets have different scales
    )

    # If we only have one target, axes might be 2D in shape (2 x n_cols).
    # If multiple targets, axes is (2*n_targets x n_cols).
    # Let's ensure we can consistently index it:
    # axes[row, col] where row = 2 * i_target or 2*i_target+1
    if n_targets == 1:
        # for convenience, always treat 'axes' as 2D
        axes = axes.reshape((2, n_cols))

    drawstyle = 'default'  # or 'steps'

    # Loop over each target, then each forecast window
    for t_idx, target_key in enumerate(targets):

        # figure out the row offset for main/residual plots of this target
        row_top = 2 * t_idx
        row_bottom = row_top + 1

        # Construct the relevant column names for actual/fitted
        actual_col = f'{target_key}_actual'
        fitted_col = f'{target_key}_fitted'
        lower_col = f'{target_key}_lower'
        upper_col = f'{target_key}_upper'

        for col_idx in range(n_cols):
            ax_top = axes[row_top, col_idx]   if n_targets > 1 else axes[0, col_idx]
            ax_bottom = axes[row_bottom, col_idx] if n_targets > 1 else axes[1, col_idx]

            # Keep track whether we've already plotted the 'Actual' for that subplot
            actual_plotted = False

            # Go through each task (i.e., each model, etc.)
            for task in tasks:
                # Pull out the lists for naming, colors, etc.
                # We assume they are parallel to 'targets'
                # i.e. task['names'][t_idx] is the name for this target
                model_name = task['names'][t_idx] if 'names' in task else target_key
                color = task['colors'][t_idx] if 'colors' in task else task['color']
                ci_alpha = task['ci_alphas'][t_idx] if 'ci_alphas' in task else task['ci_alpha']
                lw = task['lws'][t_idx] if 'lws' in task else task['lw']

                # 'results' and 'forecast' are inside task[target_key]
                n_past_windows = len(task[target_key]['results'])
                # If col_idx < n_past_windows => Past forecast i
                # elif col_idx == n_past_windows => Current forecast
                # else => no data
                if col_idx < n_past_windows:
                    df = task[target_key]['results'][col_idx]
                    errs = task[target_key]['metrics'][col_idx]
                elif col_idx == n_past_windows:
                    df = task[target_key]['forecast']
                    errs = task[target_key]['metrics'][col_idx]
                else:
                    continue  # no data for this task in this column

                # Plot the actual series only once per subplot
                if (not actual_plotted) and (col_idx != n_cols - 1):
                    if actual_col in df.columns:
                        ax_top.plot(
                            df.index, df[actual_col],
                            label='Actual',
                            color='black',
                            drawstyle=drawstyle,
                            lw=1.5
                        )
                        actual_plotted = True

                # Build a label with metrics
                label_for_legend = model_name + ' '
                if errs is not None and col_idx != n_cols - 1:
                    # Past forecast (show the "window" metrics)
                    label_for_legend += (rf"RMSE={errs['rmse']:.1f}, "
                                         rf"sMAPE={errs['smape']:.1f}")
                else:
                    # final column => 'averaged' metrics?
                    label_for_legend += (rf"$\langle$RMSE$\rangle$={errs['rmse']:.1f}, "
                                         rf"$\langle$sMAPE$\rangle$={errs['smape']:.1f}")

                # Plot fitted forecast
                if fitted_col in df.columns:
                    ax_top.plot(
                        df.index, df[fitted_col],
                        label=label_for_legend,
                        color=color,
                        drawstyle=drawstyle,
                        lw=lw
                    )

                # Confidence intervals if present
                if (lower_col in df.columns) and (upper_col in df.columns):
                    ax_top.fill_between(
                        df.index,
                        df[lower_col],
                        df[upper_col],
                        color=color,
                        alpha=ci_alpha
                    )

                # Plot residuals in bottom panel
                if actual_col in df.columns and fitted_col in df.columns:
                    residuals = df[actual_col] - df[fitted_col]
                    ax_bottom.plot(
                        df.index,
                        residuals,
                        label=model_name,
                        color=color,
                        drawstyle=drawstyle,
                        lw=lw
                    )

                # Possibly limit y-scale
                if 'ylim0' in kwargs:
                    ax_top.set_ylim(kwargs['ylim0'][0], kwargs['ylim0'][1])
                if 'ylim1' in kwargs:
                    ax_bottom.set_ylim(kwargs['ylim1'][0], kwargs['ylim1'][1])

            # Add horizontal line y=0 to residual plot
            ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=1)

            # Title for top row
            if t_idx == 0:
                if col_idx < (n_cols - 1):
                    ax_top.set_title(f'Past Forecast {max_n_results - col_idx}', fontsize=11, weight='bold')
                else:
                    ax_top.set_title('Current Forecast', fontsize=11, weight='bold')

            # Add y-labels if provided (and only on left-most columns)
            if col_idx == 0 and t_idx < len(ylabels):
                ax_top.set_ylabel(ylabels[t_idx])
                # Example for residual label
                ax_bottom.set_ylabel('Residual')

            # Legend in residual panel for the last column
            if col_idx == n_cols - 1:
                ax_bottom.legend(loc='upper left', ncol=1, fontsize=9)
            ax_top.legend(loc='lower left', ncol=1, fontsize=9)

            # Styling
            for ax in [ax_top, ax_bottom]:
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='x', direction='in', bottom=True)
                ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)

            # Improve x-axis formatting (on bottom panel)
            ax_bottom.set_xlabel('Date (month-day)', fontsize=10)
            ax_bottom.xaxis.set_major_locator(mdates.DayLocator())
            ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    fig.autofmt_xdate(rotation=45)

    # Build some name for the output
    # (e.g. combine all model names or just use run_label)
    targets = "_".join([target for target in targets])
    # models =


    plt.savefig(f"{run_label}_{'tmp'}.png", bbox_inches='tight')
    plt.show()
