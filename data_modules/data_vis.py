import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_time_series_with_residuals(tasks: list[dict], target: str = 'total_grid_load', ylabel: str = '', **kwargs):
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
            name = task['name']
            color = task['color']
            ci_alpha=task['ci_alpha']
            lw=task['lw']
            # Determine if the task has data for this column
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
    plt.savefig(f'{target}_{model_names}.png', bbox_inches='tight')
    plt.show()
