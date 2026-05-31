
// ----------------------------------------------
// Global placeholders & state
// ----------------------------------------------
var stackedChartState = {};


// Example toggleDescription function for the "Details" checkbox:
function toggleDescription() {
  // Placeholder that you can adapt to your logic
  console.log("toggleDescription was triggered");
}


// ----------------------------------------------
// 1) “Energy Mix” chart definitions
// ----------------------------------------------
const energyMixData = [
  {
    id: 1,
    title: "Energy Mix",
    dataKey: "energy_mix",
    descriptionFile: "energy_mix", // name of JSON/MD file with chart notes or description
    buttons: ["50hz", "tenn", "tran", "ampr"] // TSO area checkboxes to show
  },
];

// This function will build the <details> ... block for each item in energyMixData:
function generateEnergyMixSection({ id, title, dataKey, descriptionFile, buttons = [] }) {
  const tsoButtonsHtml = buttons.map(btnKey => {
    const btn = TSO_BUTTONS[btnKey];
    return `
      <input
        type="checkbox"
        name="tso-area"
        id="${btnKey}-checkbox-${id}"
        onchange="updateStackedChart${id}()" />
      <label for="${btnKey}-checkbox-${id}" class="${btn.colorClass}">${btn.label}</label>
    `;
  }).join("");

  // Mandatory buttons that are always shown:
  const mandatoryButtons = `
    <!-- Always show 'Total' -->
    <input type="checkbox" name="tso-area" id="total-checkbox-${id}" checked onchange="updateStackedChart${id}()" />
    <label for="total-checkbox-${id}" class="btn-purple">Total</label>

    <!-- Always show 'CI' -->
    <input type="checkbox" name="tso-area" id="showci_checkbox-${id}" onchange="updateStackedChart${id}()" />
    <label for="showci_checkbox-${id}" class="btn-purple">CI</label>

    <!-- Always show 'Details' -->
    <input type="checkbox" id="description${id}-toggle-checkbox" class="description-toggle-checkbox" onchange="toggleDescription()" />
    <label for="description${id}-toggle-checkbox" class="description-button">Details</label>

    <!-- Always show 'RESET' -->
    <label for="reloadStackedChart${id}" class="btn-purple">RESET</label>
    <input type="checkbox" id="reloadStackedChart${id}" style="display: none;" onchange="renderOrReloadChart${id}()" />
  `;

  return `
    <details class="energy-mix" open>
      <summary class="energy-mix-summary" data-i18n="${dataKey}">
        ${title}
      </summary>
      <div class="stackedChart-container" id="stackedChart${id}"></div>
      <div id="error-message${id}" class="error-message"></div>
      <div class="control-area">
        <div class="controls">
          <div class="slider-container">
            <label for="past-data-slider-${id}">Historic Data:</label>
            <input
              type="range"
              id="past-data-slider-${id}"
              min="1"
              max="100"
              step="1"
              value="20"
              onchange="updateStackedChart${id}()"
            />
          </div>
          <div class="controls-buttons">
            ${tsoButtonsHtml}
            ${mandatoryButtons}
          </div>
        </div>
      </div>
      <div id="stackedChart${id}-description-container" class="dropdown-content">
        <!-- content loaded asynchronously, e.g. via fetch for descriptionFile -->
      </div>
    </details>
  `;
}

// Insert all figures of this kind into #energy-mix
document.getElementById("energy-mix").innerHTML =
  energyMixData.map(generateEnergyMixSection).join("");

// ----------------------------------------------
// 2) Define the chart config for ID=1
// ----------------------------------------------
function getStackedChart1Config() {
  return {
    stackedChartInstance: stackedChartState["stackedChartInstance1"],
    yAxisLabel: 'Power (MW)',

    regionConfigs: [
      {
        checkboxId: 'ampr-checkbox-1',
        variables: [
          'wind_onshore_ampr','wind_offshore_ampr',
          'solar_ampr','gas_ampr','hard_coal_ampr','lignite_ampr','renewables_ampr'
        ],
        var_label: 'energy_mix_ampr',
        alias: 'Amprion',
        color: tsoColorMap['Amprion']
      },
      {
        checkboxId: 'tran-checkbox-1',
        variables: [
          'wind_onshore_tran','wind_offshore_tran',
          'solar_tran','gas_tran','hard_coal_tran','lignite_tran','renewables_tran'
        ],
        var_label: 'energy_mix_tran',
        alias: 'TransnetBW',
        color: tsoColorMap['TransnetBW']
      },
      {
        checkboxId: '50hz-checkbox-1',
        variables: [
          'wind_onshore_50hz','wind_offshore_50hz',
          'solar_50hz','gas_50hz','hard_coal_50hz','lignite_50hz','renewables_50hz'
        ],
        var_label: 'energy_mix_50hz',
        alias: '50Hertz',
        color: tsoColorMap['50Hertz']
      },
      {
        checkboxId: 'tenn-checkbox-1',
        variables: [
          'wind_onshore_tenn','wind_offshore_tenn',
          'solar_tenn','gas_tenn','hard_coal_tenn','lignite_tenn','renewables_tenn'
        ],
        var_label: 'energy_mix_tenn',
        alias: 'TenneT',
        color: tsoColorMap['TenneT']
      },
      {
        checkboxId: 'total-checkbox-1',
        variables: [
          'wind_onshore','wind_offshore',
          'solar','gas','hard_coal','lignite','renewables'
        ],
        var_label: 'energy_mix',
        alias: 'Total',
        color: tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-1',
    showIntervalId: 'showci_checkbox-1',
    errorElementId: 'error-message1',
    isDarkMode
  };
}

// ----------------------------------------------
// 3) Chart config definitions array
//    Make sure to use getStackedChart1Config
// ----------------------------------------------
const StackedChartConfigs = [
  {
    stackedChartNum: 1,
    descriptionToggleId: 'description1-toggle-checkbox',
    descriptionContainerId: 'stackedChart1-description-container',
    descLoadedKey: 'stackedChart1DescLoaded',
    createdKey: 'stackedChart1Created',
    instanceKey: 'stackedChartInstance1',
    detailsSelector: 'details.energy-mix:nth-of-type(1)',
    filePrefix: 'energy_mix_notes',
    // IMPORTANT: reference the correct function here:
    getConfigFunction: getStackedChart1Config
  }
];

// ----------------------------------------------
// 4) Setup the <details> toggles, create the chart if needed
// ----------------------------------------------
StackedChartConfigs.forEach(cfg => setupStackedChartEvents(cfg));

function setupStackedChartEvents({
  chartNum,
  descriptionToggleId,
  descriptionContainerId,
  descLoadedKey,
  createdKey,
  instanceKey,
  detailsSelector,
  filePrefix,
  getConfigFunction
}) {
  // 6.1) “Details” toggles the chart creation
  document
    .querySelector(detailsSelector)
    .addEventListener('toggle', async function(e) {
      // Only create the chart when user opens the details for the first time
      if (e.target.open && !stackedChartState[createdKey]) {
        stackedChartState[createdKey] = true;
        stackedChartState[instanceKey] = await createStackedChart(`#stackedChart${stackedChartNum}`, getBaseStackedChartOptions());
        window[`updateStackedChart${stackedChartNum}`](); // first update
      }
    });

  // 6.2) “Details” button (checkbox) for loading the MD file
  document
    .getElementById(descriptionToggleId)?.addEventListener('click', async function() {
      const content = document.getElementById(descriptionContainerId);
      const isVisible = (content.style.display === 'block');
      content.style.display = isVisible ? 'none' : 'block';

      // Load only once
      if (!isVisible && !stackedChartState[descLoadedKey]) {
        stackedChartState[descLoadedKey] = true;
        const language = 'en'; // or from i18n
        const fileName = `${filePrefix}_${language}.md`;
        // Example loadMarkdown call
        await loadMarkdown(`data/forecasts/${fileName}`, descriptionContainerId);
      }
    });

  // 6.3) Global function to destroy & recreate
  window[`renderOrReloadChart${stackedChartNum}`] = async function() {
    if (stackedChartState[instanceKey]) {
      stackedChartState[instanceKey].destroy();
      stackedChartState[createdKey] = false;
    }
    stackedChartState[createdKey] = true;
    stackedChartState[instanceKey] = await createStackedChart(`#stackedChart${stackedChartNum}`, getBaseStackedChartOptions());
    window[`updateStackedChart${stackedChartNum}`]();
  };

  // 6.4) Global function to update chart
  window[`updateStackedChart${stackedChartNum}`] = async function() {
    const config = getConfigFunction();
    await updateStackedChartGeneric(config, stackedChartNum);
  };
}

// Example stub for loading Markdown (optional)
async function loadMarkdown(url, containerId) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to load Markdown file');
    const text = await response.text();
    document.getElementById(containerId).innerText = text;
  } catch (err) {
    console.error(err);
  }
}

// ----------------------------------------------
// 5) Create the base ApexCharts “stacked area” chart
// ----------------------------------------------
function getBaseStackedChartOptions() {
  return {
    stackedChart: {
      type: 'area',
      height: 350,
      stacked: true,
      toolbar: { show: true },
      zoom: { enabled: true }
    },
    dataLabels: { enabled: false },
    stroke: {
      curve: 'smooth',
      width: 2
    },
    xaxis: {
      type: 'datetime'
    },
    yaxis: {
      labels: {
        formatter: val => val.toFixed(0)
      }
    },
    tooltip: {
      x: {
        format: 'dd MMM HH:mm'
      }
    },
    series: []
  };
}

async function createStackedChart(selector, baseOptions) {
  const stackedChart = new ApexCharts(document.querySelector(selector), baseOptions);
  await stackedChart.render();
  return stackedChart;
}

// ----------------------------------------------
// 6) Fetch & merge data from the three JSON files for a label
// ----------------------------------------------
const forecastCache = {}; // optional caching to avoid re-fetching

async function fetchAllForecastData(varLabel) {
  // Return from cache if available
  if (forecastCache[varLabel]) {
    return forecastCache[varLabel];
  }
  const basePath = `./data/forecasts/${varLabel}`;
  const fileNames = [
    'forecast_prev_actual.json',
    'forecast_prev_fitted.json',
    'forecast_curr_fitted.json'
  ];

  // The final structure must be an array of {name, data:[ [timestamp, value], ... ]}.
  // We'll merge by “name”.
  const allSeriesMap = {}; // key = name, value = array of [time, val]

  for (let f of fileNames) {
    try {
      const res = await fetch(`${basePath}/${f}`);
      if (!res.ok) throw new Error(`Cannot fetch ${basePath}/${f}`);
      const jsonData = await res.json();
      // jsonData should be like [{ name, data: [[ts, val], ...]}, ...]
      for (let seriesObj of jsonData) {
        const sName = seriesObj.name;
        if (!allSeriesMap[sName]) {
          allSeriesMap[sName] = [];
        }
        // Append all [time, val] pairs
        allSeriesMap[sName].push(...seriesObj.data);
      }
    } catch (err) {
      console.error("Error fetching", f, "for", varLabel, err);
      // You could show an error, or just continue
    }
  }

  // Convert map -> array of {name, data}
  const mergedArray = Object.entries(allSeriesMap).map(([sName, pairs]) => {
    // Sort by timestamp if needed
    pairs.sort((a, b) => new Date(a[0]) - new Date(b[0]));
    return { name: sName, data: pairs };
  });

  // Cache it
  forecastCache[varLabel] = mergedArray;
  return mergedArray;
}

// ----------------------------------------------
// 7) The generic “updateStackedChartGeneric” function
// ----------------------------------------------
async function updateStackedChartGeneric(config, stackedChartNum) {
  const stackedChart = config.stackedChartInstance;
  if (!stackedChart) return; // Chart not created yet

  const errorEl = document.getElementById(config.errorElementId);
  if (errorEl) errorEl.textContent = ""; // clear old errors

  // 1) Past data slider
  const sliderVal = document.getElementById(`past-data-slider-${stackedChartNum}`)?.value ?? 20;
  // The slider could represent hours, or a percentage, or something else.
  // For this example, we’ll just keep it as a variable you might use for slicing or filtering.

  // 2) For each regionConfig, if the corresponding checkbox is checked, fetch & merge data
  let finalSeries = [];

  for (const regionCfg of config.regionConfigs) {
    const checkBox = document.getElementById(regionCfg.checkboxId);
    if (!checkBox) continue;
    if (!checkBox.checked) continue;

    try {
      const dataArr = await fetchAllForecastData(regionCfg.var_label);
      // dataArr is an array of objects like { name, data:[ [ts, val], ... ] }

      // Keep only variables from regionCfg.variables
      const subset = dataArr.filter(seriesObj => regionCfg.variables.includes(seriesObj.name));

      // Possibly rename or do other transformations:
      for (let sObj of subset) {
        // e.g. rename "wind_onshore_ampr" to "Amprion: wind_onshore"
        const shortName = sObj.name.replace(`_${regionCfg.alias.toLowerCase()}`, '');
        const newSeriesName = regionCfg.alias + ": " + shortName;

        // (Optional) further slicing by time or amount. For simplicity, we keep all:
        finalSeries.push({
          name: newSeriesName,
          data: sObj.data
        });
      }
    } catch (err) {
      console.error(err);
      if (errorEl) {
        errorEl.textContent = "Error loading data: " + err.message;
      }
    }
  }

  // 3) Update the chart with new theme & data
  stackedChart.updateOptions({
    theme: {
      mode: config.isDarkMode ? 'dark' : 'light'
    },
    yaxis: {
      title: {
        text: config.yAxisLabel
      }
    }
  });

  stackedChart.updateSeries(finalSeries);
}