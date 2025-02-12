// GLOBAL DEFINITIONS
let baseUrl = "https://raw.githubusercontent.com/vsevolodnedora/energy_market_analysis/main/deploy/";



/**
 * Show only one content section and hide the others.
 * @param {string} sectionId - The ID of the section to display.
 */



/**
 * Toggles a subpage’s visibility by adding/removing an .active class.
 * Called by the onClick of each checkbox in the top nav.
 *
 * @param {string} subpageId - The ID of the subpage container div.
 * @param {boolean} isChecked - true if checkbox is checked (show), false if unchecked (hide).
 */
function toggleSubpage(subpageId, isChecked) {
  const subpage = document.getElementById(subpageId);
  if (!subpage) return;
  if (isChecked) {
    subpage.classList.add('active');
  } else {
    subpage.classList.remove('active');
  }



}
/************************************************************
 * 0) Utils
 ************************************************************/

// Example color utility
function lightenColor(color, percent) {
  const num = parseInt(color.slice(1), 16),
      amt = Math.round(2.55 * percent),
      R = (num >> 16) + amt,
      G = (num >> 8 & 0x00FF) + amt,
      B = (num & 0x0000FF) + amt;
      return `#${(
              0x1000000 +
              (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x25000 +
              (G < 255 ? (G < 1 ? 0 : G) : 255) * 0x250 +
              (B < 255 ? (B < 1 ? 0 : B) : 255)
          ).toString(16).slice(1).toUpperCase()
      }`;
}

/************************************************************
 * 0) Language
 ************************************************************/

// Update all elements with [data-i18n] using i18next
function updateContent() {
  document.querySelectorAll('[data-i18n]').forEach(element => {
      const key = element.getAttribute('data-i18n');
      element.innerHTML = i18next.t(key);
  });
}

// Function to load JSON file asynchronously
async function loadTranslations(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load translations from ${url}`);
    }
    return await response.json();
}


// Initialize i18next with external resources
async function initializeI18n() {
    try {
        const resources = await loadTranslations('translations.json');

        // Initialize i18next with loaded resources
        await i18next.init({
            lng: 'en', // default language
            debug: false,
            resources: resources
        });

        updateContent();

    } catch (error) {
        console.error('Error initializing i18next:', error);
    }
}

// Toggle between English and German
async function toggleLanguage() {
    const newLang = (i18next.language === 'en') ? 'de' : 'en';
    await i18next.changeLanguage(newLang);

    updateContent(); // Updates text translations

    if (chartState["chartInstance1"]) {
        updateChart1(); // Force chart update to reformat labels/axes
    }
    if (chartState["chartInstance2"]) {
        updateChart2();
    }
    if (chartState["chartInstance3"]) {
        updateChart3();
    }
    // Reload the description in the new language if already loaded
    if (chartState["chart1DescLoaded"]) {
        const language = i18next.language; // Get the new current language
        const fileName = `wind_offshore_notes_${language}.md`  ;
        await loadMarkdown(`data/forecasts/${fileName}`, 'chart1-description-container');
    }


    // Reload HTML files with different languages (1/3)
    const mainInfoFileName = (newLang === 'en') ? 'main_info_en.html' : 'main_info_de.html';
    await loadHTML(`${mainInfoFileName}`, 'main_info-content');

    // Reload HTML files with different languages (1/3)
    const apiInfoFileName = (newLang === 'en') ? 'api_info_en.html' : 'api_info_de.html';
    await loadHTML(`${apiInfoFileName}`, 'api_info-content');

    // Update the text of the language toggle button
    const languageToggleButton = document.getElementById('language-toggle');
    languageToggleButton.textContent = (newLang === 'en') ? '🌍 DE' : '🌍 EN'; // Show the other language

}


/************************************************************
 * 0) HTML LOADERS (LANGUAGE DEPENDENT) (2 and 3 / 3)
 ************************************************************/

document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Load the default mainFile content based on the initial language
        const initialLanguage = i18next.language || 'en'; // Use 'en' if not set
        const mainFileFileName = (initialLanguage === 'en') ? 'main_info_en.html' : 'main_info_de.html';
        await loadHTML(`${mainFileFileName}`, 'main_info-content');
    } catch (error) {
        console.error('Error initializing i18next or loading mainFile content:', error);
    }
});
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Load the default apiFile content based on the initial language
        const initialLanguage = i18next.language || 'en'; // Use 'en' if not set
        const apiFileFileName = (initialLanguage === 'en') ? 'api_info_en.html' : 'api_info_de.html';
        await loadHTML(`${apiFileFileName}`, 'api_info-content');
    } catch (error) {
        console.error('Error initializing i18next or loading apiFile content:', error);
    }
});


// Function to dynamically load HTML content into a target container
async function loadHTML(filePath, containerId) {
    try {
        const response = await fetch(filePath);
        if (!response.ok) throw new Error(`Failed to load ${filePath}`);
        const htmlContent = await response.text();
        const container = document.getElementById(containerId);
        container.innerHTML = htmlContent;
    } catch (error) {
        console.error(`Error loading HTML content from ${filePath}:`, error);
        const container = document.getElementById(containerId);
        container.innerHTML = '<p>Error loading content. </p>';
    }

    Prism.highlightAll(); // syntax highlighting
}

/************************************************************
 * 0) Dark Mode
 ************************************************************/

let isDarkMode = true;


let chartState = {
   "chart1DescLoaded": false, // offshore wind power
   "chart2DescLoaded": false, // onshore wind power
   "chart3DescLoaded": false, // solar power

   "chart1Created": false,
   "chart2Created": false,
   "chart3Created": false,

   "chartInstance1": null,
   "chartInstance2": null,
   "chartInstance3": null
};

/************************************************************
 * 0) Load Markdown FIles
 ************************************************************/

// Helper function to load Markdown from a given URL
async function loadMarkdown(url, containerId) {
    const fallbackUrl = baseUrl + url;
    try {
        // Attempt to fetch the file from the local path
        let response = await fetch(url);

        // If the response is not OK, throw an error to trigger the fallback
        if (!response.ok) {
            console.warn(`Failed to load markdown from local path: ${url}. Trying fallback URL.`);
            response = await fetch(fallbackUrl);
        }

        // If the fallback response is also not OK, throw an error
        if (!response.ok) {
            throw new Error(`Failed to load markdown from both local and fallback URLs.`);
        }

        const markdownText = await response.text();

        // Use showdown to convert the markdown to HTML
        const converter = new showdown.Converter({
            tables: true,
            ghCompatibleHeaderId: true,
            simplifiedAutoLink: true,
            strikethrough: true,
            tasklists: true,
            emoji: true,
            parseImgDimensions: true,
            openLinksInNewWindow: true,
            simpleLineBreaks: true
        });
        const html = converter.makeHtml(markdownText);

        // Insert HTML into the container
            document.getElementById(containerId).innerHTML = html;
    } catch (error) {
        console.error(error);
        document.getElementById(containerId).innerHTML = `
            <p style="color:red;">
                <strong>Error:</strong> Could not load description.
            </p>`;
    }
}


// Optional: start in dark mode
toggleDarkMode();

/************************************************************
 * -1) Create a CACHE
 ************************************************************/
// Global cache to store data once fetched
const forecastDataCache = {};

/************************************************************
 * 0) Fetches a data file and returns it as an array of { x: Date, y: number }.
 ************************************************************/
async function getCachedData(variable, file, errorElementId) {
  const locDir = 'data/forecasts';  // local directory
  const cacheKey = `${variable}-${file}`;

  // If data is already in cache, return immediately
  if (forecastDataCache[cacheKey]) {
    return forecastDataCache[cacheKey];
  }

  // Otherwise, fetch from default location
  try {
    const response = await fetch(`${locDir}/${variable}/${file}`);
    if (!response.ok) {
      throw new Error(`Failed to load ${variable} from default location`);
    }
    const data = await response.json();
    forecastDataCache[cacheKey] = data.map(([t, v]) => ({ x: new Date(t), y: v }));
    return forecastDataCache[cacheKey];
  } catch (error) {
    console.warn(error.message);
  }

  // Attempt fallback only if default fetch fails
  try {
    const fallbackResponse = await fetch(`${baseUrl}${locDir}/${variable}/${file}`);
    if (!fallbackResponse.ok) {
      throw new Error(`Failed to load ${variable} from fallback URL`);
    }
    const fallbackData = await fallbackResponse.json();
    forecastDataCache[cacheKey] = fallbackData.map(([t, v]) => ({ x: new Date(t), y: v }));
    return forecastDataCache[cacheKey];
  } catch (fallbackError) {
    console.error(fallbackError.message);
    document.getElementById(errorElementId).textContent = fallbackError.message;
    // Return null if both attempts fail
    forecastDataCache[cacheKey] = null;
    return null;
  }
}

/************************************************************
 * 1) Create a new chart in a given container
 ************************************************************/
async function createChart(containerSelector, baseOptions) {
  const chart = new ApexCharts(
    document.querySelector(containerSelector), baseOptions
  );
  await chart.render();
  return chart;
}

/************************************************************
 * 3) Function that adds series (and intervals) to the chart
 ************************************************************/
async function addSeries({
  variable,
  alias,
  color,
  pastDataRatio,
  seriesData,
  annotations,
  errorElementId
}) {
  // Standard file names
  const prevFittedFile = 'forecast_prev_fitted.json';
  const prevActualFile = 'forecast_prev_actual.json';
  const currFittedFile = 'forecast_curr_fitted.json';

  // Fetch data in parallel
  const [
    pastFittedData,
    pastActualData,
    currentData
  ] = await Promise.all([
    getCachedData(variable, prevFittedFile, errorElementId),
    getCachedData(variable, prevActualFile, errorElementId),
    getCachedData(variable, currFittedFile, errorElementId)
  ]);

  // -------------------- PAST FITTED (Solid Line) --------------------
  if (pastFittedData && pastFittedData.length > 0) {
    const pastToShow = Math.floor(pastFittedData.length * pastDataRatio);
    seriesData.push({
      name : `${alias} (${i18next.t('past-fitted-label')})`,
      data : pastFittedData.slice(-pastToShow),
      color: color,
      type : 'line',
      stroke: {
        width: 2,
        dashArray: 0,
        curve: 'smooth'
      }
    });
  }

  // -------------------- PAST ACTUAL (Dashed Line) --------------------
  if (pastActualData && pastActualData.length > 0) {
    const pastToShow = Math.floor(pastActualData.length * pastDataRatio);
    seriesData.push({
      name : `${alias} (${i18next.t('past-actual-label')})`,
      data : pastActualData.slice(-pastToShow),
      color: color,//lightenColor(color, 20),
      stroke: {
        width: 2,
        dashArray: 5,
        curve: 'smooth'
      }
    });
  }

  // -------------------- CURRENT FORECAST (Solid Line) --------------------
  if (currentData && currentData.length > 0) {
    seriesData.push({
      name : `${alias} (${i18next.t('current-label')})`,
      data : currentData,
      color: color,
      type : 'line'
    });

    // Annotation for the first forecast point
    const lastForecastTime = currentData[0].x.getTime();
    annotations.push({
      x: lastForecastTime,
      label: {
        text: i18next.t('last-forecast-label'),
        style: { color: '#FFFFFF', background: '#808080' }
      }
    });

    // Add vertical lines going back until the beginning of prevFittedFile
    if (pastFittedData && pastFittedData.length > 0) {
      const forecastDuration = currentData[currentData.length - 1].x.getTime() - lastForecastTime;
      let newAnnotationTime = lastForecastTime;

      while (newAnnotationTime > pastFittedData[0].x.getTime()) {
        newAnnotationTime -= forecastDuration;
        annotations.push({
          x: newAnnotationTime,
          label: {
//            text: i18next.t('new-vertical-line-label'),
            style: { color: '#FFFFFF', background: '#FF0000' }
          }
        });
      }
    }
  }
}

/************************************************************
 * Function that adds confidence intervals (area regions) to the chart
 ************************************************************/
 async function addCI({
  variable,
  alias,
  color,
  showInterval,
  pastDataRatio,
  seriesData,
  errorElementId
}) {
  // Standard file names
  const prevLowerFile = 'forecast_prev_lower.json';
  const prevUpperFile = 'forecast_prev_upper.json';
  const currLowerFile = 'forecast_curr_lower.json';
  const currUpperFile = 'forecast_curr_upper.json';

  // Fetch data in parallel
  const [
    pastLowerData,
    pastUpperData,
    currentLowerData,
    currentUpperData
  ] = await Promise.all([
    getCachedData(variable, prevLowerFile, errorElementId),
    getCachedData(variable, prevUpperFile, errorElementId),
    getCachedData(variable, currLowerFile, errorElementId),
    getCachedData(variable, currUpperFile, errorElementId)
  ]);

  // -------------------- PREV FORECAST INTERVAL (Area) --------------------
  if (showInterval && pastLowerData && pastUpperData) {
    if (pastLowerData.length === pastUpperData.length && pastLowerData.length > 0) {
      const pastLength = Math.floor(pastLowerData.length * pastDataRatio);
      const lowerSlice = pastLowerData.slice(-pastLength);
      const upperSlice = pastUpperData.slice(-pastLength);

      const pastForecastPolygon = [
        ...lowerSlice.map((pt) => ({ x: pt.x, y: pt.y })),
        ...upperSlice.slice().reverse().map((pt) => ({ x: pt.x, y: pt.y }))
      ];

      if (pastForecastPolygon.length > 0) {
        seriesData.push({
          name        : `${alias} (${i18next.t('prev-forecast-interval-label')})`,
          type        : 'area',
          data        : pastForecastPolygon,
          color       : color,
          fillOpacity : 0.1
        });
      }
    }
  }

  // -------------------- CURRENT FORECAST INTERVAL (Area) --------------------
  if (showInterval && currentLowerData && currentUpperData) {
    if (currentLowerData.length === currentUpperData.length && currentLowerData.length > 0) {
      const forecastPolygon = [
        ...currentLowerData.map((pt) => ({ x: pt.x, y: pt.y })),
        ...currentUpperData.slice().reverse().map((pt) => ({ x: pt.x, y: pt.y }))
      ];

      if (forecastPolygon.length > 0) {
        seriesData.push({
          name : `${alias} (${i18next.t('forecast-interval-label')})`,
          type : 'area',
          data : forecastPolygon,
          color: color,
          fillOpacity : 0.1
        });
      }
    }
  }
}

/************************************************************
 * 4) The generic “updateChart” Pass a config object so you can re-use for onshore, solar, etc.
 ************************************************************/
async function updateChartGeneric(config) {
  const {
    chartInstance,
    yAxisLabel,
    regionConfigs,
    pastDataSliderId,
    showIntervalId,
    errorElementId,
    isDarkMode
  } = config;

  // If the chart is not yet created, do nothing
  if (!chartInstance) return;

  // Clear old errors
  document.getElementById(errorElementId).textContent = '';

  // Prepare arrays for data
  const seriesData = [];
  const annotations = [];

  // Get user preferences from the DOM
  const pastDataRatio = document.getElementById(pastDataSliderId).value / 100;
  const showInterval = document.getElementById(showIntervalId).checked;

  // Fetch and build series data for each selected region
  for (const region of regionConfigs) {
    const checkbox = document.getElementById(region.checkboxId);
    if (checkbox && checkbox.checked) {
      // Fetch series for the region
      await addSeries({
        variable: region.variable,
        alias: region.alias,
        color: region.color,
        pastDataRatio: pastDataRatio,
        seriesData: seriesData,
        annotations: annotations,
        errorElementId:errorElementId
      });
    }
  }

  // attempt to split to remove artifacts from turning CI off
  for (const region of regionConfigs) {
    const checkbox = document.getElementById(region.checkboxId);
    if (checkbox && checkbox.checked) {
      // Fetch confidence intervals for the region if showInterval is enabled
      if (showInterval) {
        await addCI({
          variable: region.variable,
          alias: region.alias,
          color: region.color,
          showInterval: showInterval,
          pastDataRatio: pastDataRatio,
          seriesData: seriesData,
          errorElementId: errorElementId
        });
      }
    }
  }

  // Ensure no leftover CI data remains in seriesData
  const filteredSeriesData = seriesData.filter(series => {
    // Remove past CI series when `showInterval` is false
    if (!showInterval && series.name.includes(i18next.t('prev-forecast-interval-label'))) {
      return false;
    }
    return true;
  });

  // Add a “Now” line annotation
  const now = new Date();
  annotations.push({
    x: now.getTime(),
    borderColor: '#FF0000',
    label: {
      text: i18next.t('now-label'),
      style: { color: '#FFF', background: '#FF0000' }
    }
  });

  // Update the chart with filtered data and annotations
  chartInstance.updateOptions({
    series: filteredSeriesData,
    annotations: {
      xaxis: annotations,
      yaxis: [],
      points: [],
      texts: [
        {
          x: '3%',
          y: '3%',
          text: yAxisLabel,
          borderColor: 'transparent',
          style: {
            fontSize: '14px',
            color: isDarkMode ? '#e0e0e0' : '#000',
            fontWeight: 'bold',
          },
        },
      ],
    },
    stroke: {
      width: 1, // Set line width
      dashArray: Array(regionConfigs.length).fill([3, 0, 3]).flat(), // Dynamically set dashArray
    },

    tooltip: {
        theme: isDarkMode ? 'dark' : 'light' ,
        format: 'dd MMM HH:mm', // e.g., "05 Feb 14:00"
    },
    xaxis: {
      labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } },
      title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } },
    },
    yaxis: {
      labels: {
        style: {
          colors: isDarkMode ? '#e0e0e0' : '#000',
          fontSize: '14px',
        },
        formatter: function (value) {
          return Math.round(value);
        },
      },
      tickAmount: 5,
      min: 0,
      forceNiceScale: true,
    },
    // legend: {
    //   labels: { show:false, colors: isDarkMode ? '#e0e0e0' : '#000' },
    // },
    chart: {
      zoom: {
        enabled: true,
        type: 'xy',
      },

    },

    legend: {
      show: true,
      position: 'top', // Position of the legend
      horizontalAlign: 'center', // Align legend in the center
      offsetY: 20, // Adjust vertical position (positive values move it lower)
      formatter: function (seriesName, opts) {
        // Limit to 3 items
        const index = opts.seriesIndex;
        if (index < 3) {
          return seriesName; // Show the series name for the first 3 items
        }
        return ''; // Hide the legend item for the rest
      },
    },

  });

}

/************************************************************
 * 5) Setup for the first chart
 ************************************************************/
function getBaseChartOptions() {
  return {
      chart: {
          type: 'line',
          height: 350,

          toolbar: { show: true }
      },
      series: [{stroke:{dashArray: 5}}], // Add your series data here
      // makers: [],
      // lines: [],
      xaxis: {
          type: 'datetime',
          labels: {
              style: { colors: isDarkMode ? '#e0e0e0' : '#000' },
              formatter: function (val, timestamp) {
                  const currentLang = i18next.language;
                  const dateFormatter = new Intl.DateTimeFormat(currentLang, {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      // minute: '2-digit',
                      // hour12: false,
                  });
                  return dateFormatter.format( new Date(timestamp) );
              }
          },
          title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }
      },
      yaxis: {
        title: {
//            text: 'MW'//i18next.t('offshore-power-label'),
            offsetX: 300, // Move the label far to the right
            offsetY: -50, // Move the label to the top
            style: {
                color   : isDarkMode ? '#e0e0e0' : '#000',
                fontSize: '14px', // Adjust this size as needed
            },
        },
        labels: {
            style: {
                colors  : isDarkMode ? '#e0e0e0' : '#000',
                fontSize: '14px', // Adjust this size as needed
            },
            formatter: function(value) {
                return Math.round(value); // Format as integers
            },
        },
        // tickAmount: 5, // Optional: control the number of ticks on the Y-axis
        // forceNiceScale: true, // Optional: ensure nice intervals on Y-axis
      },
      annotations: { xaxis: [] },
      tooltip: {
          shared: true, // Ensure the tooltip is shared across all series
          intersect: false, // Trigger tooltip for all points at the X-coordinate
          theme: isDarkMode ? 'dark' : 'light',
          x: {
              format: 'dd MMM yyyy HH:mm'
          },
          y: {
              formatter: function (value, { series, seriesIndex, dataPointIndex, w }) {
                  return value !== null ? value.toFixed(2) : 'N/A'; // Customize formatting
              }
          }
      },
      legend: {
          labels: {
              colors: isDarkMode ? '#e0e0e0' : '#000',
              useSeriesColors: false
          }
      }
  };
}

/************************************************************
 * 6) Controls
 ************************************************************/
const forecastData = [
  {
    id: 1,
    title: "Offshore Wind Power Forecast",
    dataKey: "offshore-forecast",
    descriptionFile: "wind_offshore_notes",
    // Show all TSO areas for this ID:
    buttons: ["50hz", "tenn"]
  },
  {
    id: 2,
    title: "Onshore Wind Power Forecast",
    dataKey: "onshore-forecast",
    descriptionFile: "wind_onshore_notes",
    // Only show 50Hertz & TenneT for this ID:
    buttons: ["50hz", "tenn", "tran", "ampr"]
  },
  {
    id: 3,
    title: "Solar Power Forecast",
    dataKey: "solar-forecast",
    descriptionFile: "solar_notes",
    // Show none of the TSO checkboxes here (only the "always" buttons):
    buttons: ["50hz", "tenn", "tran", "ampr"]
  },
  {
    id: 4,
    title: "Load Forecast",
    dataKey: "load-forecast",
    descriptionFile: "load_notes",
    // Show none of the TSO checkboxes here (only the "always" buttons):
    buttons: ["50hz", "tenn", "tran", "ampr"]
  },
  {
    id: 5,
    title: "Generation Forecast",
    dataKey: "generation-forecast",
    descriptionFile: "generation_notes",
    // Show none of the TSO checkboxes here (only the "always" buttons):
    buttons: ["50hz", "tenn", "tran", "ampr"]
  },
];

// Helper object to define each TSO button’s label & CSS class
const TSO_BUTTONS = {
  "50hz": { label: "50Hertz", colorClass: "btn-blue" },
  "tenn": { label: "TenneT", colorClass: "btn-green" },
  "tran": { label: "TransnetBW", colorClass: "btn-red" },
  "ampr": { label: "Amprion", colorClass: "btn-yellow" }
};

function generateForecastSection({ id, title, dataKey, descriptionFile, buttons = [] }) {
  // Build the HTML for any TSO buttons this forecast wants:
  const tsoButtonsHtml = buttons.map(btnKey => {
    const btn = TSO_BUTTONS[btnKey];
    return `
      <input type="checkbox" name="tso-area" id="${btnKey}-checkbox-${id}" onchange="updateChart${id}()" />
      <label for="${btnKey}-checkbox-${id}" class="${btn.colorClass}">${btn.label}</label>
    `;
  }).join("");

  // Mandatory buttons that are always shown
  const mandatoryButtons = `
    <!-- Always show 'Total' -->
    <input type="checkbox" name="tso-area" id="total-checkbox-${id}" checked onchange="updateChart${id}()" />
    <label for="total-checkbox-${id}" class="btn-purple">Total</label>

    <!-- Always show 'CI' -->
    <input type="checkbox" name="tso-area" id="showci_checkbox-${id}" onchange="updateChart${id}()" />
    <label for="showci_checkbox-${id}" class="btn-purple">CI</label>

    <!-- Always show 'Details' -->
    <input type="checkbox" id="description${id}-toggle-checkbox" class="description-toggle-checkbox" onchange="toggleDescription()" />
    <label for="description${id}-toggle-checkbox" class="description-button" data-i18n="details-label">Details</label>

    <!-- Always show 'RESET' -->
    <label for="reloadChart${id}" class="btn-purple">RESET</label>
    <input type="checkbox" id="reloadChart${id}" style="display: none;" onchange="renderOrReloadChart${id}()" />
  `;

  return `
    <details class="forecast-section" open>
      <summary class="forecast-summary" data-i18n="${dataKey}">
        ${title}
      </summary>
      <div class="chart-container" id="chart${id}"></div>
      <div id="error-message${id}" class="error-message"></div>
      <div class="control-area">
        <div class="controls">
          <div class="slider-container">
            <label for="past-data-slider-${id}" data-i18n="historic-data">Historic Data:</label>
            <input
              type="range"
              id="past-data-slider-${id}"
              min="1"
              max="100"
              step="1"
              value="20"
              onchange="updateChart${id}()"
            />
          </div>
          <div class="controls-buttons">
            ${tsoButtonsHtml}
            ${mandatoryButtons}
          </div>
        </div>
      </div>
      <div id="chart${id}-description-container" class="dropdown-content">
        <!-- content loaded asynchronously, e.g. via fetch for descriptionFile -->
      </div>
    </details>
  `;
}

// Insert all forecast sections into the page
document.getElementById("individual-forecasts").innerHTML = forecastData
  .map(generateForecastSection)
  .join("");

const tsoColorMap = {
  "50Hertz": "#0000FF",  // Blue
  "TenneT": "#008000",   // Green
  "TransnetBW": "#FF0000", // Red
  "Amprion": "#FFFF00",  // Yellow
  "Total": "#800090"     // Purple
};

/************************************************************
 * 6.0) Define an array describing each chart
 ************************************************************/

const getChart1Config = () => {
  return {
    chartInstance   : chartState["chartInstance1"],
    yAxisLabel      : 'Power (MW)',//i18next.t('offshore-power-label-mw'),

    regionConfigs   : [
      {
        checkboxId: '50hz-checkbox-1',
        variable  : 'wind_offshore_50hz',
        alias     : '50Hertz',
        color     : tsoColorMap['50Hertz'],
      },
      {
        checkboxId: 'tenn-checkbox-1',
        variable  : 'wind_offshore_tenn',
        alias     : 'TenneT',
        color     : tsoColorMap['TenneT'],
      },
      {
        checkboxId: 'total-checkbox-1',
        variable  : 'wind_offshore',
        alias     : 'Total',
        color     : tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-1',
    showIntervalId  : 'showci_checkbox-1',
    errorElementId  : 'error-message1',
    isDarkMode      : isDarkMode // or define it yourself
  };
};

const getChart2Config = () => {
  return {
    chartInstance   : chartState["chartInstance2"],
    yAxisLabel      : 'Power (MW)',//i18next.t('onshore-power-label-mw'),

    regionConfigs   : [
      {
        checkboxId: 'ampr-checkbox-2',
        variable  : 'wind_onshore_ampr',
        alias     : 'Amprion',
        color     : tsoColorMap['Amprion'],
      },
      {
        checkboxId: 'tran-checkbox-2',
        variable  : 'wind_onshore_tran',
        alias     : 'TransnetBW',
        color     : tsoColorMap['TransnetBW'],
      },
      {
        checkboxId: '50hz-checkbox-2',
        variable  : 'wind_onshore_50hz',
        alias     : '50Hertz',
        color     : tsoColorMap['50Hertz'],
      },
      {
        checkboxId: 'tenn-checkbox-2',
        variable  : 'wind_onshore_tenn',
        alias     : 'TenneT',
        color     : tsoColorMap['TenneT'],
      },
      {
        checkboxId: 'total-checkbox-2',
        variable  : 'wind_onshore',
        alias     : 'Total',
        color     : tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-2',
    showIntervalId  : 'showci_checkbox-2',
    errorElementId  : 'error-message2',
    isDarkMode      : isDarkMode // or define it yourself
  };
};

const getChart3Config = () => {
  return {
    chartInstance   : chartState["chartInstance3"],
    yAxisLabel      : 'Power (MW)',//i18next.t('onshore-power-label-mw'),

    regionConfigs   : [
      {
        checkboxId: 'ampr-checkbox-3',
        variable  : 'solar_ampr',
        alias     : 'Amprion',
        color     : tsoColorMap['Amprion'],
      },
      {
        checkboxId: 'tran-checkbox-3',
        variable  : 'solar_tran',
        alias     : 'TransnetBW',
        color     : tsoColorMap['TransnetBW'],
      },
      {
        checkboxId: '50hz-checkbox-3',
        variable  : 'solar_50hz',
        alias     : '50Hertz',
        color     : tsoColorMap['50Hertz'],
      },
      {
        checkboxId: 'tenn-checkbox-3',
        variable  : 'solar_tenn',
        alias     : 'TenneT',
        color     : tsoColorMap['TenneT'],
      },
      {
        checkboxId: 'total-checkbox-3',
        variable  : 'solar',
        alias     : 'Total',
        color     : tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-3',
    showIntervalId  : 'showci_checkbox-3',
    errorElementId  : 'error-message3',
    isDarkMode      : isDarkMode // or define it yourself
  };
};

const getChart4Config = () => {
  return {
    chartInstance   : chartState["chartInstance4"],
    yAxisLabel      : 'Load (MW)',//i18next.t('onshore-power-label-mw'),

    regionConfigs   : [
      {
        checkboxId: 'ampr-checkbox-4',
        variable  : 'load_ampr',
        alias     : 'Amprion',
        color     : tsoColorMap['Amprion'],
      },
      {
        checkboxId: 'tran-checkbox-4',
        variable  : 'load_tran',
        alias     : 'TransnetBW',
        color     : tsoColorMap['TransnetBW'],
      },
      {
        checkboxId: '50hz-checkbox-4',
        variable  : 'load_50hz',
        alias     : '50Hertz',
        color     : tsoColorMap['50Hertz'],
      },
      {
        checkboxId: 'tenn-checkbox-4',
        variable  : 'load_tenn',
        alias     : 'TenneT',
        color     : tsoColorMap['TenneT'],
      },
      {
        checkboxId: 'total-checkbox-4',
        variable  : 'load',
        alias     : 'Total',
        color     : tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-4',
    showIntervalId  : 'showci_checkbox-4',
    errorElementId  : 'error-message4',
    isDarkMode      : isDarkMode // or define it yourself
  };
};

const getChart5Config = () => {
  return {
    chartInstance   : chartState["chartInstance5"],
    yAxisLabel      : 'Generation (MW)',//i18next.t('onshore-power-label-mw'),

    regionConfigs   : [
      {
        checkboxId: 'ampr-checkbox-5',
        variable  : 'generation_ampr',
        alias     : 'Amprion',
        color     : tsoColorMap['Amprion'],
      },
      {
        checkboxId: 'tran-checkbox-5',
        variable  : 'generation_tran',
        alias     : 'TransnetBW',
        color     : tsoColorMap['TransnetBW'],
      },
      {
        checkboxId: '50hz-checkbox-5',
        variable  : 'generation_50hz',
        alias     : '50Hertz',
        color     : tsoColorMap['50Hertz'],
      },
      {
        checkboxId: 'tenn-checkbox-5',
        variable  : 'generation_tenn',
        alias     : 'TenneT',
        color     : tsoColorMap['TenneT'],
      },
      {
        checkboxId: 'total-checkbox-5',
        variable  : 'generation',
        alias     : 'Total',
        color     : tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-5',
    showIntervalId  : 'showci_checkbox-5',
    errorElementId  : 'error-message5',
    isDarkMode      : isDarkMode // or define it yourself
  };
};

const chartConfigs = [
  {
    chartNum: 1,
    descriptionToggleId: 'description1-toggle-checkbox',
    descriptionContainerId: 'chart1-description-container',
    descLoadedKey: 'chart1DescLoaded',
    createdKey: 'chart1Created',
    instanceKey: 'chartInstance1',
    detailsSelector: 'details:nth-of-type(1)',
    filePrefix: 'wind_offshore_notes',
    getConfigFunction: getChart1Config
  },
  {
    chartNum: 2,
    descriptionToggleId: 'description2-toggle-checkbox',
    descriptionContainerId: 'chart2-description-container',
    descLoadedKey: 'chart2DescLoaded',
    createdKey: 'chart2Created',
    instanceKey: 'chartInstance2',
    detailsSelector: 'details:nth-of-type(1)',
    filePrefix: 'wind_onshore_notes',
    getConfigFunction: getChart2Config
  },
  {
    chartNum: 3,
    descriptionToggleId: 'description3-toggle-checkbox',
    descriptionContainerId: 'chart3-description-container',
    descLoadedKey: 'chart3DescLoaded',
    createdKey: 'chart3Created',
    instanceKey: 'chartInstance3',
    detailsSelector: 'details:nth-of-type(1)',
    filePrefix: 'solar_notes',
    getConfigFunction: getChart3Config
  },
  {
    chartNum: 4,
    descriptionToggleId: 'description4-toggle-checkbox',
    descriptionContainerId: 'chart4-description-container',
    descLoadedKey: 'chart4DescLoaded',
    createdKey: 'chart4Created',
    instanceKey: 'chartInstance4',
    detailsSelector: 'details:nth-of-type(1)',
    filePrefix: 'load_notes',
    getConfigFunction: getChart4Config
  },
  {
    chartNum: 5,
    descriptionToggleId: 'description5-toggle-checkbox',
    descriptionContainerId: 'chart5-description-container',
    descLoadedKey: 'chart5DescLoaded',
    createdKey: 'chart5Created',
    instanceKey: 'chartInstance5',
    detailsSelector: 'details:nth-of-type(1)',
    filePrefix: 'generation_notes',
    getConfigFunction: getChart5Config
  }
];

/************************************************************
 * 6.1) The actual update function for “Offshore” Chart #1
 *    (matching the onChange handlers in the HTML)
 ************************************************************/

function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
  isDarkMode = !isDarkMode;

  // If charts exist, refresh them (loop over all instances)
  // Example: any key named "chartInstanceX" in chartState
  for (let key of Object.keys(chartState)) {
    if (key.startsWith('chartInstance') && chartState[key]) {
      // Extract the chart number from the key, e.g. "chartInstance1" -> "1"
      const chartNum = key.replace('chartInstance', '');
      // Call updateChart1(), updateChart2(), ...
      window[`updateChart${chartNum}`]?.();
    }
  }

  window["updateStackedChart100"]?.();
}

/************************************************************
 * 6.2) Helper function to set up each chart’s event listeners
 *    and “render/reload” + “update” functions.
 ************************************************************/
function setupChartEvents({
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
  // 6.2a) Toggle the Markdown description
  document
    .getElementById(descriptionToggleId)
    .addEventListener('click', async function () {
      const content = document.getElementById(descriptionContainerId);

      // Toggle visibility
      const isVisible = (content.style.display === 'block');
      content.style.display = isVisible ? 'none' : 'block';

      // If opening it for the first time, load the Markdown
      if (!isVisible && !chartState[descLoadedKey]) {
        chartState[descLoadedKey] = true;

        // Determine the language and load
        const language = i18next.language; // e.g. 'en' or 'de'
        const fileName = `${filePrefix}_${language}.md`;
        await loadMarkdown(`data/forecasts/${fileName}`, descriptionContainerId);
      }
    });

  // 6.2b) <details> toggle to create the chart only when opened
  document
    .querySelector(detailsSelector)
    .addEventListener('toggle', async function(e) {
      if (e.target.open && !chartState[createdKey]) {
        await initializeI18n();   // loads i18n, sets default language
        chartState[createdKey] = true;
        chartState[instanceKey] = await createChart(`#chart${chartNum}`, getBaseChartOptions());
        window[`updateChart${chartNum}`](); // first update
      }
    });

  // 6.2c) Create a global function like “renderOrReloadChart1”
  //     but parametric for each chartNum
  window[`renderOrReloadChart${chartNum}`] = async function() {
    // If the chart exists, destroy it
    if (chartState[instanceKey]) {
      chartState[instanceKey].destroy();
      chartState[createdKey] = false;
    }
    // Recreate
    await initializeI18n();
    chartState[createdKey] = true;
    chartState[instanceKey] = await createChart(`#chart${chartNum}`, getBaseChartOptions());
    window[`updateChart${chartNum}`](); // first update
  };

  // 6.2d) Create a global function like “updateChart1” for each chartNum
  window[`updateChart${chartNum}`] = async function() {
    const config = getConfigFunction(); // e.g. getChart1Config()
    await updateChartGeneric(config);
  };
}

/************************************************************
 * 6.3) Loop through our array and set everything up in one go
 ************************************************************/
chartConfigs.forEach(cfg => setupChartEvents(cfg));




/************************************************************
 * ========================================================= *
 ************************************************************/





/********************************************
 * 2) “Energy Mix” chart definitions & HTML
 ********************************************/

var stackedChartState = {};

/********************************************
 * 2) “Energy Mix” chart definitions & HTML
 ********************************************/

const energyMixData = [
  {
    id: 100,
    title: "Energy Mix",
    dataKey: "energy_mix",
    descriptionFile: "energy_mix", // JSON/MD file name for your notes
    buttons: ["50hz", "tenn", "tran", "ampr"] // TSO area checkboxes to show
  },
];

// make sure that buttons 'undo' previous button
function toggleExclusiveSelection(checkbox) {
  // Get all checkboxes with name "tso-area" in the same section
  const checkboxes = checkbox.closest('.control-area').querySelectorAll('input[name="tso-area"]');
  checkboxes.forEach(cb => {
    if (cb !== checkbox) {
      cb.checked = false;
    }
  });
}

// This function builds the <details> ... block for each item in energyMixData:
function generateEnergyMixSection({ id, title, dataKey, descriptionFile, buttons = [] }) {
  const tsoButtonsHtml = buttons.map(btnKey => {
    const btn = TSO_BUTTONS[btnKey];  // Ensure TSO_BUTTONS is globally defined
    return `
      <input
        type="checkbox"
        name="tso-area"
        id="${btnKey}-checkbox-${id}"
        onchange="toggleExclusiveSelection(this); updateStackedChart${id}()" />
      <label for="${btnKey}-checkbox-${id}" class="btn-purple">${btn.label}</label>
    `;
  }).join("");

  // Mandatory buttons that are always shown:
  const mandatoryButtons = `
    <!-- 'Total' button with exclusive selection -->
    <input type="checkbox" name="tso-area" id="total-checkbox-${id}" checked
      onchange="toggleExclusiveSelection(this); updateStackedChart${id}()" />
    <label for="total-checkbox-${id}" class="btn-purple">Total</label>

    <!-- 'Details' -->
    <input type="checkbox" id="description${id}-toggle-checkbox" class="description-toggle-checkbox" />
    <label for="description${id}-toggle-checkbox" class="description-button">Details</label>

    <!-- 'RESET' button -->
    <label for="reloadStackedChart${id}" class="btn-purple">RESET</label>
    <input type="checkbox" id="reloadStackedChart${id}" style="display: none;" onchange="renderOrReloadChart${id}()" />
  `;

  return `
    <details class="energy-mix" open>
      <summary class="energy-mix-summary" data-i18n="${dataKey}">
        ${title}
      </summary>
      <div class="chart-stack-container">
        <div class="lineChart2-container" id="lineChart2${id}-totalLine"></div>
        <div class="lineChart-container" id="lineChart${id}-totalLine"></div>
        <div class="stackedChart-container" id="stackedChart${id}"></div>
      </div>
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
// Insert all figures into #energy-mix
document.getElementById("energy-mix").innerHTML =
  energyMixData.map(generateEnergyMixSection).join("");

/*****************************************************
 * 3) Define the chart config for the chart ID=100
 *****************************************************/

// Define fixed color mapping to ensure all contributions are visible
const energyMixColorMapping = {
   'wind_onshore': '#00008B',  // Dark Blue TODO make a bit lighter (natural blue)
   'wind_offshore': '#ADD8E6', // Light Blue
   'solar': '#FFD700',         // Gold
   'gas': '#D3D3D3',           // Light Gray (make slightly darker -- normal gray)
   'hard_coal': '#000000',     // Black
   'lignite': '#8B4513',       // Darker Brown
   'renewables': '#008000',    // Green
   'biomass': '#90EE90',       // Light Green
   'oil': '#A9A9A9',           // Dark Gray
   'waste': '#D2B48C',         // Light Brown (Tan)
   'other_fossil': '#A52A2A',  // Brown
   'coal_derived_gas':'#6B4423'// brown-gray
};

const stackedChartSegmentOrder = new Map([
    ['wind_onshore', 0],
    ['wind_offshore', 1],
    ['solar', 2],
    ['renewables', 3],  // Hydro & other renewables
    ['biomass', 4],
    ['waste', 5],
    ['lignite', 6],
    ['hard_coal', 7],
    ['coal_derived_gas', 8],
    ['gas', 9],
    ['oil', 10],
    ['other_fossil', 11]
]);


const nameMapping = {
    'wind_onshore': 'Wind Onshore',
    'wind_offshore': 'Wind Offshore',
    'solar': 'Solar',
    'gas': 'Fossil Gas',
    'hard_coal': 'Hard Coal',
    'lignite': 'Lignite',
    'renewables': 'Renewables',
    'biomass': 'Biomass',
    'oil': 'Oil',
    'waste': 'Waste',
    'other_fossil': 'Other Fossil Fuels',
    'coal_derived_gas':"Coal-derived Gas"
};


function getStackedChart100Config() {
  return {
    stackedChartInstance: stackedChartState["stackedChartInstance100"],
    yAxisLabel: 'Power (MW)',
    chartId: 100,
    regionConfigs: [
      {
        checkboxId: 'ampr-checkbox-100',
        variables: [
          'wind_onshore','wind_offshore', 'solar',
          'gas','hard_coal','lignite','renewables'
        ],
        var_label: 'energy_mix_ampr',  // The folder name for this TSO
        generation_var_label: 'generation_ampr',
        load_var_label: 'load_ampr',
        carbon_var_label: 'carbon_intensity_ampr',
        alias: 'Amprion',
        color: tsoColorMap['Amprion']  // Make sure tsoColorMap is defined globally
      },
      {
        checkboxId: 'tran-checkbox-100',
        variables: [
          'wind_onshore','wind_offshore', 'solar',
          'gas','hard_coal','lignite','renewables'
        ],
        var_label: 'energy_mix_tran',
        generation_var_label: 'generation_tran',
        load_var_label: 'load_tran',
        carbon_var_label: 'carbon_intensity_tran',
        alias: 'TransnetBW',
        color: tsoColorMap['TransnetBW']
      },
      {
        checkboxId: '50hz-checkbox-100',
        variables: [
          'wind_onshore','wind_offshore', 'solar',
          'gas','hard_coal','lignite','renewables'
        ],
        var_label: 'energy_mix_50hz',
        generation_var_label: 'generation_50hz',
        load_var_label: 'load_50hz',
        carbon_var_label: 'carbon_intensity_50hz',
        alias: '50Hertz',
        color: tsoColorMap['50Hertz']
      },
      {
        checkboxId: 'tenn-checkbox-100',
        variables: [
          'wind_onshore','wind_offshore', 'solar',
          'gas','hard_coal','lignite','renewables'
        ],
        var_label: 'energy_mix_tenn',
        generation_var_label: 'generation_tenn',
        load_var_label: 'load_tenn',
        carbon_var_label: 'carbon_intensity_tenn',
        alias: 'TenneT',
        color: tsoColorMap['TenneT']
      },
      {
        checkboxId: 'total-checkbox-100',
        variables: [
          'wind_onshore','wind_offshore', 'solar',
          'gas','hard_coal','lignite','renewables'
        ],
        var_label: 'energy_mix',  // “Total” folder name
        generation_var_label: 'generation',
        load_var_label: 'load',
        carbon_var_label: 'carbon_intensity',
        alias: 'Total',
        color: tsoColorMap['Total']
      }
    ],

    pastDataSliderId: 'past-data-slider-100',
    showIntervalId: 'showci_checkbox-100',
    errorElementId: 'error-message100',
    isDarkMode      : isDarkMode // or define it yourself
  };
}

const StackedChartConfigs = [
  {
    stackedChartNum: 100,
    descriptionToggleId: 'description100-toggle-checkbox',
    descriptionContainerId: 'stackedChart100-description-container',
    descLoadedKey: 'stackedChart100DescLoaded',
    createdKey: 'stackedChart100Created',
    instanceKey: 'stackedChartInstance100',
    detailsSelector: 'details.energy-mix:nth-of-type(1)',
    filePrefix: 'energy_mix_notes',
    getConfigFunction: getStackedChart100Config
  }
];

/***************************************************
 * 6) Fetch & merge data from the three JSON files
 ***************************************************/

// Cache for forecast data
const forecastCache = {};

async function fetchForecastData(varLabel, fileName) {
    /**
     * Fetches forecast data from a specified file and processes it into a structured time-series format.
     * @param {string} varLabel - The variable label representing the dataset type.
     * @param {string} fileName - The filename of the forecast data.
     * @returns {Array} - An array of processed series objects with sorted time-series data.
     */
    const basePath = `./data/forecasts/${varLabel}`;
    const allSeriesMap = {}; // Stores the fetched data categorized by series name

    try {
        console.log(`Fetching: ${basePath}/${fileName}`);
        const res = await fetch(`${basePath}/${fileName}`);
        if (!res.ok) throw new Error(`Cannot fetch ${basePath}/${fileName}`);
        const jsonData = await res.json();
        console.log(`Data fetched for ${fileName}:`, jsonData);

        // Process each series object in the fetched JSON data
        for (let seriesObj of jsonData) {
            const sName = seriesObj.name; // Extract series name
            if (!allSeriesMap[sName]) {
                allSeriesMap[sName] = []; // Initialize array if it does not exist
            }

            // Convert timestamp strings to Date objects and validate them
            for (let pair of seriesObj.data) {
                const date = new Date(pair[0]);
                if (isNaN(date.getTime())) {
                    console.warn(`Invalid date encountered: ${pair[0]} in ${fileName}`);
                    continue; // Skip invalid date entries
                }
                allSeriesMap[sName].push([date, pair[1]]);
            }
        }
    } catch (err) {
        console.error("Error fetching", fileName, "for", varLabel, err);
    }

    // Convert the processed data into an array of objects formatted for plotting
    return Object.entries(allSeriesMap).map(([sName, pairs]) => {
        // Sort data points by time to maintain chronological order
        pairs.sort((a, b) => a[0] - b[0]);

        // Convert Date objects back to ISO strings for consistency
        return { name: sName, data: pairs.map(([date, value]) => [date.toISOString(), value]) };
    });
}

async function processStackedChartData(config) {
    let finalSeries = {}; // Store merged data by variable name
    let forecastStartTime = null; // Variable to store the forecast start time
    const pastDataRatio = document.getElementById(config.pastDataSliderId)?.value / 100 || 1;

    for (const regionCfg of config.regionConfigs) {
        const checkBox = document.getElementById(regionCfg.checkboxId);
        if (!checkBox || !checkBox.checked) continue; // Skip unchecked regions

        try {
            console.log(`Processing region: ${regionCfg.var_label}`);

            // Load both current fitted and past actual forecast datasets concurrently
            const [currFittedData, prevActualData] = await Promise.all([
                fetchForecastData(regionCfg.var_label, 'forecast_curr_fitted.json'),
                fetchForecastData(regionCfg.var_label, 'forecast_prev_actual.json'),
//                fetchForecastData(regionCfg.var_label, 'forecast_prev_fitted.json')
            ]);

            // Determine forecast start time by checking the last timestamp in past actual data
            if (prevActualData.length > 0) {
                const lastEntry = prevActualData[prevActualData.length - 1];
                if (lastEntry.data.length > 0) {
                    forecastStartTime = new Date(lastEntry.data[lastEntry.data.length - 1][0]).getTime();
                }
            }

            const mergedData = {}; // Temporary storage for merged datasets

            function mergeData(data, isPastData = false) {
                for (let sObj of data) {
                    // Extract a short variable name by removing the region alias suffix
                    const shortName = sObj.name.replace(`_${regionCfg.alias.toLowerCase()}`, '');
                    if (!mergedData[shortName]) mergedData[shortName] = [];

                    // Convert timestamps to JavaScript Date objects for proper sorting and plotting
                    let seriesData = sObj.data.map(([timestamp, value]) => [new Date(timestamp).getTime(), value]);

                    if (isPastData) {
                        // Limit the past data based on user-defined past data ratio
                        const pastToShow = Math.floor(seriesData.length * pastDataRatio);
                        seriesData = seriesData.slice(-pastToShow);
                    }

                    // Append processed data to the merged dataset
                    mergedData[shortName].push(...seriesData);
                }
            }
            // Merge current and past data into a unified structure
            mergeData(currFittedData, false);
            mergeData(prevActualData, true);

            // Convert merged dataset into final series format suitable for plotting
            for (let shortName in mergedData) {
                finalSeries[shortName] = {
                    name: nameMapping[shortName],//`${regionCfg.alias}: ${shortName}`, // Format name for legend
                    data: mergedData[shortName].sort((a, b) => a[0] - b[0]), // Ensure chronological order
                    color: energyMixColorMapping[shortName] || '#555' // Assign color or default gray
                };
            }
        } catch (err) {
            console.error(err); // Log errors without interrupting execution
        }
    }

    return { finalSeries: Object.values(finalSeries), forecastStartTime }; // Return structured time-series data and forecast reference point
}

function computeTotalSeries(finalSeries) {
  /**
   * finalSeries is an array of { name, data: [[ts, val], [ts, val], ...], color: ... }
   * We want to sum across all series at each timestamp => produce [[ts, totalVal], ...].
   */
  const sumMap = new Map();  // key = timestamp (numeric), val = sum of all MW

  // Go through each series, accumulate the value in sumMap
  for (const seriesObj of finalSeries) {
    for (const [tsStr, val] of seriesObj.data) {
      // Convert ISO string to numeric time to unify
      const t = new Date(tsStr).getTime();
      sumMap.set(t, (sumMap.get(t) || 0) + (val || 0));
    }
  }

  // Convert map -> sorted array of [timestamp, totalVal]
  const summedData = Array.from(sumMap.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([t, totalVal]) => [new Date(t).toISOString(), totalVal]);

  // Return an Apex-compatible single series
  return [{
    name: 'Total Generation (Sum)',
    data: summedData,
    color: '#FF0000' // pick a highlight color for the line
  }];
}

/*****************************************************
 * 7) Plot the data
 *****************************************************/

function getBaseLineChartOptions() {
  return {
    chart: {
      type: 'line',
      height: 200,   // you can choose a smaller height since it's “on top”
      toolbar: { show: false }, // Hides toolbar controls
      zoom: { enabled: false } // Disables zooming feature
    },
    dataLabels: { enabled: false },
    stroke: {
      curve: 'smooth',
      width: 2
    },
    xaxis: {
      type: 'datetime',
      labels: {
        show: false,
        style: { colors: isDarkMode ? '#e0e0e0' : '#000' }
      }, // Hides x-axis labels
      title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }
    },
    yaxis: {
      min: 0,
      title: {
        text: 'Power (MW)',
        style: { color: isDarkMode ? '#e0e0e0' : '#000' },
        fontSize: '16px'
      },
      labels: {
        style: { colors: isDarkMode ? '#e0e0e0' : '#000' },
        formatter: val => val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val.toFixed(0),
        fontSize: '16px'
      }
    },
    tooltip: {
      theme: isDarkMode ? 'dark' : 'light',
      x: { format: 'dd MMM HH:mm' }
    },
    grid: {
      show: true,
      borderColor: isDarkMode ? '#444' : '#E0E0E0', // Adjust grid color based on mode
      strokeDashArray: 3, // Dashed lines for a thin appearance
      xaxis: {
        lines: { show: true } // Enable vertical grid lines
      },
      yaxis: {
        lines: { show: true } // Enable horizontal grid lines
      }
    },
    legend: {
      labels: {
//        colors: isDarkMode ? '#e0e0e0' : '#000',
        useSeriesColors: false
      }
    },
    // We will manually set the series with update
    series: []
  };
}

async function createTotalLineChart(selector, baseOptions) {
  const chart = new ApexCharts(document.querySelector(selector), baseOptions);
  await chart.render();
  return chart;
}
async function createTotalLineChart2(selector, baseOptions) {
  const chart = new ApexCharts(document.querySelector(selector), baseOptions);
  await chart.render();
  return chart;
}

async function updateTotalLineChart(
    config, generationActualSeries, generationSeries, loadActualSeries, loadSeries, forecastStartTime
) {
  const lineChart = stackedChartState[`lineChartInstance${config.chartId}`];
  if (!lineChart) return;

  const isDarkMode = config.isDarkMode;

  // Convert actual and fitted series to timestamp format
  const formattedLoadActualSeries = loadActualSeries.map(series => ({
    ...series,
    name: 'Load',
    color: '#FF0000',
    data: series.data.map(({ x, y }) => [new Date(x).getTime(), y])
  }));

  const formattedLoadSeries = loadSeries.map(series => ({
    ...series,
    name: 'Load',
    color: '#FF0000',
    data: series.data.map(({ x, y }) => [new Date(x).getTime(), y])
  }));

  const formattedGenerationActualSeries = generationActualSeries.map(series => ({
    ...series,
    name: 'Generation',
    color: '#00CC00',
    data: series.data.map(({ x, y }) => [new Date(x).getTime(), y])
  }));

  const formattedGenerationSeries = generationSeries.map(series => ({
    ...series,
    name: 'Generation',
    color: '#00CC00',
    data: series.data.map(({ x, y }) => [new Date(x).getTime(), y])
  }));

  const finalSeries = [];

  [...formattedGenerationActualSeries, ...formattedLoadActualSeries].forEach(series => {
    finalSeries.push({
      name: series.name + " (Actual)",
      data: series.data,
      color: series.color,
      stroke: {
        width: 2,
        dashArray: 0 // Solid line
      }
    });
  });

  [...formattedGenerationSeries, ...formattedLoadSeries].forEach(series => {
    finalSeries.push({
      name: series.name + " (Forecast)",
      data: series.data,
      color: series.color,
      stroke: {
        width: 2,
        dashArray: [5, 3] // Dashed line
      }
    });
  });

  // Find the minimum value from all data points
  const allYValues = finalSeries.flatMap(series => series.data.map(point => point[1]));
  const minYValue = Math.min(...allYValues);

  // Clear old data & update series
  lineChart.updateSeries([]);
  lineChart.updateSeries(finalSeries);

  // Update chart options
    lineChart.updateOptions({
      theme: { mode: isDarkMode ? 'dark' : 'light' },
      annotations: { xaxis: getForecastAnnotations(forecastStartTime, false) },
      stroke: {
        width: 2,
        dashArray: finalSeries.map(series =>
          series.name.includes("Forecast") ? [5, 3] : 0
        )
      },
      xaxis: {
        type: 'datetime',
        labels: {
          style: {
            colors: isDarkMode ? '#e0e0e0' : '#000',
            fontSize: '12px'  // Correctly set font size
          },
          formatter: function (val, timestamp) {
            const currentLang = i18next.language;
            const dateFormatter = new Intl.DateTimeFormat(currentLang, {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
            });
            return dateFormatter.format(new Date(timestamp));
          }
        }
      },
      yaxis: {
        min: minYValue,
        title: {
          text: config.yAxisLabel,
          style: { fontSize: '12px' } // Correct syntax for title font size
        },
        labels: {
          formatter: val => val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val.toFixed(1),
          style: {  // Correct placement inside style
            colors: isDarkMode ? '#e0e0e0' : '#000',
            fontSize: '12px'
          }
        }
      },
      legend: {
        position: 'top',
        horizontalAlign: 'center',
        floating: true,
        markers: { width: 10, height: 10 },
        itemMargin: { horizontal: 10, vertical: 5 },
        formatter: function(seriesName, opts) {
          const seriesIndex = opts.seriesIndex;
          return seriesIndex < 2 ? seriesName : null;
        }
      }
    });
}

async function updateTotalLineChart2(
    config, loadActualSeries, loadFittedSeries, forecastStartTime
) {
  const lineChart2 = stackedChartState[`lineChart2Instance${config.chartId}`];
  if (!lineChart2) return;

  const isDarkMode = config.isDarkMode;

  // Convert each data point x into a timestamp for both actual and fitted series
  const formattedLoadActualSeries = loadActualSeries.map(series => ({
    ...series,
    data: series.data.map(({ x, y }) => [new Date(x).getTime(), y])
  }));

  const formattedLoadFittedSeries = loadFittedSeries.map(series => ({
    ...series,
    data: series.data.map(({ x, y }) => [new Date(x).getTime(), y])
  }));

  // Calculate the minimum y value dynamically
  const allYValues = [
    ...formattedLoadActualSeries.flatMap(series => series.data.map(point => point[1])),
    ...formattedLoadFittedSeries.flatMap(series => series.data.map(point => point[1]))
  ];
  const minYValue = Math.min(...allYValues);

  // Ensure same color for both actual and fitted series
  const color = isDarkMode ? '#999999' : '#000000';

  const finalSeries = [
    {
      name: 'Carbon Intensity (Actual)',
      data: formattedLoadActualSeries.flatMap(series => series.data),
      color: color,
      stroke: {
        width: 2,
        dashArray: 0 // Solid line
      }
    },
    {
      name: 'Carbon Intensity (Forecast)',
      data: formattedLoadFittedSeries.flatMap(series => series.data),
      color: color,
      stroke: {
        width: 2,
        dashArray: [5, 3] // Dashed line
      }
    }
  ];

  // Clear and set new data
  lineChart2.updateSeries([]);
  lineChart2.updateSeries(finalSeries);


    // Update chart options
    lineChart2.updateOptions({
      theme: { mode: isDarkMode ? 'dark' : 'light' },
      annotations: { xaxis: getForecastAnnotations(forecastStartTime, false) },
      stroke: {
        width: 2,
        dashArray: finalSeries.map(series =>
          series.name.includes("Forecast") ? [5, 3] : 0
        )
      },
      xaxis: {
        type: 'datetime',
        labels: {
          style: {
            colors: isDarkMode ? '#e0e0e0' : '#000',
            fontSize: '12px'  // Ensure font size is applied
          },
          formatter: function (val, timestamp) {
            const currentLang = i18next.language;
            const dateFormatter = new Intl.DateTimeFormat(currentLang, {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
            });
            return dateFormatter.format(new Date(timestamp));
          }
        }
      },
      yaxis: {
        min: minYValue,  // Dynamically set based on data
        title: {
          text: 'Carbon Intensity (gCO₂/kWh)',
          style: { fontSize: '12px' } // Corrected font size application
        },
        labels: {
          formatter: val => val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val.toFixed(1),
          style: {  // Properly applying font size
            colors: isDarkMode ? '#e0e0e0' : '#000',
            fontSize: '12px'
          }
        }
      },
      legend: {
        show: false,
        position: 'top',
        horizontalAlign: 'center',
        floating: true,
        markers: { width: 10, height: 10 },
        itemMargin: { horizontal: 10, vertical: 5 },
        formatter: function(seriesName, opts) {
          if (seriesName.includes("Forecast")) {
            return seriesName.replace(" (Forecast)", ""); // Clean legend display
          }
          return seriesName;
        }
      }
    });

}


async function createStackedChart(selector, baseOptions) {
  const chart = new ApexCharts(document.querySelector(selector), baseOptions);
  await chart.render();
  return chart;
}

function getBaseStackedChartOptions() {
  return {
    chart: {
      type: 'area',
      height: 350,
      stacked: true,
      toolbar: { show: false }, // Hides toolbar controls
      zoom: { enabled: false } // Disables zooming feature
    },
    plotOptions: {
      area: {
        stacking: 'normal' // Ensures areas are stacked on top of each other
      }
    },
    dataLabels: { enabled: false },
    stroke: {
      curve: 'smooth',
      width: 2 // Slightly reduced width for better visibility
    },
    fill: {
      type: 'solid' // Ensures solid stacking without transparency overlap
    },
    legend: {
      position: 'bottom',
      horizontalAlign: 'center',
      labels: {
//        colors: isDarkMode ? '#e0e0e0' : '#000',
        useSeriesColors: false
      }
    },
    xaxis: {
      type: 'datetime',
      labels: {
        style: { colors: isDarkMode ? '#e0e0e0' : '#000' },
        formatter: function (val, timestamp) {
          const currentLang = i18next.language;
          const dateFormatter = new Intl.DateTimeFormat(currentLang, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
          });
          return dateFormatter.format(new Date(timestamp));
        }
      },
      title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }
    },
    yaxis: {
      min: 0,
      title: {
        text: 'Power Output (MW)',
        style: {
          color: isDarkMode ? '#e0e0e0' : '#000',
          fontSize: '14px'
        }
      },
      labels: {
        style: {
          colors: isDarkMode ? '#e0e0e0' : '#000',
          fontSize: '14px'
        },
        formatter: val => val >= 1000 ? (val / 1000).toFixed(0) + 'k' : val.toFixed(0)
      }
    },
    tooltip: {
      theme: isDarkMode ? 'dark' : 'light',
      x: { format: 'dd MMM HH:mm' }
    },
    grid: {
      show: true,
      borderColor: isDarkMode ? '#555' : '#E0E0E0', // Darker grid lines in dark mode
      strokeDashArray: 3, // Dashed lines for a thin appearance
      xaxis: { lines: { show: true } }, // Enable vertical grid lines
      yaxis: { lines: { show: true } } // Enable horizontal grid lines
    },
    series: []
  };
}


function updateStackedChart(config, finalSeries, forecastStartTime) {
    /**
     * Updates the stacked area chart with processed time-series data.
     * @param {Object} config - Configuration object containing chart instance and settings.
     * @param {Array} finalSeries - Array of series objects to be plotted.
     * @param {number|null} forecastStartTime - Timestamp for forecast annotation line.
     */
    const isDarkMode = config.isDarkMode;
    const stackedChart = config.stackedChartInstance;
    if (!stackedChart) return;

    //
    finalSeries.sort((a, b) => (stackedChartSegmentOrder.get(a.name.split(": ")[1]) || 99) -
                                (stackedChartSegmentOrder.get(b.name.split(": ")[1]) || 99));


    // Update chart settings including axis labels, colors, and themes
    stackedChart.updateOptions({
        theme: { mode: isDarkMode ? 'dark' : 'light' },
        colors: Object.values(energyMixColorMapping),
        xaxis: {
            type: 'datetime',
            labels: {
                style: {
                    colors: isDarkMode ? '#e0e0e0' : '#000',
                    fontSize: '12px'  // Ensure font size is applied
                }
            },
            title: {
                text: config.xAxisLabel || '', // Ensure a title is set if needed
                style: {
                    color: isDarkMode ? '#e0e0e0' : '#000',
                    fontSize: '12px'  // Apply font size to the x-axis title
                }
            }
        },
        yaxis: {
            min: 0,
            title: {
                text: config.yAxisLabel,
                style: {
                    fontSize: '12px'  // Corrected font size application
                }
            },
            labels: {
                formatter: val => val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val.toFixed(1),
                style: {
                    colors: isDarkMode ? '#e0e0e0' : '#000',
                    fontSize: '12px'  // Correctly applying font size
                }
            }
        },
        stroke: { curve: 'smooth', width: 2 },
        fill: { type: 'solid' },
        annotations: { xaxis: getForecastAnnotations(forecastStartTime, true) }
    });


    // Clear previous series and update with new dataset
    stackedChart.updateSeries([]);
    stackedChart.updateSeries(finalSeries);
}

async function updateStackedChartGeneric(config) {
    const { finalSeries, forecastStartTime } = await processStackedChartData(config);
    updateStackedChart(config, finalSeries, forecastStartTime, isDarkMode);
}

function getForecastAnnotations(x, withText) {
    /**
     * Generates forecast annotations for charts.
     * @param {number|null} x - Timestamp for forecast annotation line.
     * @param {boolean} withText - Determines if labels should be shown.
     * @returns {Array} Annotations array for the chart.
     */
    if (!x) return [];

    const baseAnnotation = {
        x: x,
        borderColor: "#555",
        strokeDashArray: 5,
    };

    if (!withText) {
        return [baseAnnotation];
    }

    return [
        {
            ...baseAnnotation,
            label: {
                borderColor: "#777777",
                position: "top",
                offsetX: -5,
                text: "Actual",
                style: {
                    background: "#777777",
                    color: "#FFF",
                    fontSize: "12px",
                    rotate: -180,
                },
            },
        },
        {
            ...baseAnnotation,
            label: {
                borderColor: "#777777",
                position: "top",
                offsetX: 25,
                text: "Forecast",
                style: {
                    background: "#777777",
                    color: "#FFF",
                    fontSize: "12px",
                    rotate: -90,
                },
            },
        },
    ];
}


// Insert all figures into #energy-mix
//document.getElementById("energy-mix").innerHTML =
//  energyMixData.map(generateEnergyMixSection).join("");

/* --- Refactored setupStackedChartEvents and helper functions --- */

/**
 * Sets up the click event listener for the description toggle.
 * When clicked, it toggles the description container’s visibility and lazy-loads its content.
 */
function setupDescriptionToggleEvent({
    descriptionToggleId, descriptionContainerId, descLoadedKey, filePrefix }) {
  const toggleElement = document.getElementById(descriptionToggleId);
  if (!toggleElement) return;

  toggleElement.addEventListener('click', async function() {
    const content = document.getElementById(descriptionContainerId);
    const isVisible = (content.style.display === 'block');
    content.style.display = isVisible ? 'none' : 'block';

    // Lazy-load the markdown content if not loaded yet
    if (!isVisible && !stackedChartState[descLoadedKey]) {
      stackedChartState[descLoadedKey] = true;
      const language = 'en'; // Adjust if localization is used
      const fileName = `${filePrefix}_${language}.md`;
      await loadMarkdown(`data/forecasts/${fileName}`, descriptionContainerId);
    }
  });
}

/**
 * Sets up the 'toggle' event listener for the <details> element.
 * When expanded for the first time, it initializes i18n and creates the charts.
 */
function setupDetailsToggleEvent({ detailsSelector, stackedChartNum, createdKey, instanceKey }) {
  const detailsElement = document.querySelector(detailsSelector);
  if (!detailsElement) return;

  detailsElement.addEventListener('toggle', async function(e) {
    if (e.target.open && !stackedChartState[createdKey]) {
      await initializeI18n();
      stackedChartState[createdKey] = true;

      // Create the charts
      stackedChartState[instanceKey] = await createStackedChart(
        `#stackedChart${stackedChartNum}`,
        getBaseStackedChartOptions()
      );
      stackedChartState[`lineChartInstance${stackedChartNum}`] = await createTotalLineChart(
        `#lineChart${stackedChartNum}-totalLine`,
        getBaseLineChartOptions()
      );
      stackedChartState[`lineChart2Instance${stackedChartNum}`] = await createTotalLineChart2(
        `#lineChart2${stackedChartNum}-totalLine`,
        getBaseLineChartOptions()
      );
      // First update of the charts
      window[`updateStackedChart${stackedChartNum}`]();
    }
  });
}

/**
 * Defines the global reset (reload) function for the charts.
 * When called (e.g. via the RESET button), it destroys and recreates the charts.
 */
function defineResetChartFunction({ stackedChartNum, createdKey, instanceKey }) {
  window[`renderOrReloadChart${stackedChartNum}`] = async function() {
    if (stackedChartState[instanceKey]) {
      stackedChartState[instanceKey].destroy();
      stackedChartState[createdKey] = false;
    }
    if (stackedChartState[`lineChartInstance${stackedChartNum}`]) {
      stackedChartState[`lineChartInstance${stackedChartNum}`].destroy();
    }
    if (stackedChartState[`lineChart2Instance${stackedChartNum}`]) {
      stackedChartState[`lineChart2Instance${stackedChartNum}`].destroy();
    }

    await initializeI18n();
    stackedChartState[createdKey] = true;

    stackedChartState[instanceKey] = await createStackedChart(
      `#stackedChart${stackedChartNum}`,
      getBaseStackedChartOptions()
    );
    stackedChartState[`lineChartInstance${stackedChartNum}`] = await createTotalLineChart(
      `#lineChart${stackedChartNum}-totalLine`,
      getBaseLineChartOptions()
    );
    stackedChartState[`lineChart2Instance${stackedChartNum}`] = await createTotalLineChart2(
      `#lineChart2${stackedChartNum}-totalLine`,
      getBaseLineChartOptions()
    );

    window[`updateStackedChart${stackedChartNum}`]();
  };
}

/**
 * Defines the global update function for both the stacked area chart and the total line chart.
 * This function is called after initialization and whenever a control is changed.
 */
function defineUpdateChartFunction({ stackedChartNum, getConfigFunction }) {
  window[`updateStackedChart${stackedChartNum}`] = async function() {
    const config = getConfigFunction(); // Call getConfigFunction once per update
    const { finalSeries, forecastStartTime } = await processStackedChartData(config);

    let forecastStartTime_ = null; // Initialize should be the same for all files

    // --- total generation data
    let combinedGenerationActualSeries = [];
    let combinedGenerationFittedSeries = [];

    for (const regionCfg of config.regionConfigs) {
      const checkBox = document.getElementById(regionCfg.checkboxId);
      if (!checkBox || !checkBox.checked) continue;

      const series = await getCombinedLoadSeries(
        regionCfg.generation_var_label,
        'forecast_prev_actual.json',
        'forecast_prev_fitted.json',
        'forecast_curr_fitted.json',
        config
      );
      if (series) {
        combinedGenerationActualSeries.push(series.actualSeries); // Store actual separately
        combinedGenerationFittedSeries.push(series.fittedSeries); // Store fitted separately
        // Extract forecastStartTime from the last actual data point (if available)
        if (!forecastStartTime_){
            forecastStartTime_ = series.actualSeries.forecastStartTime;
        }
      }
    }
    if (combinedGenerationActualSeries.length === 0){
      const logDiv = document.getElementById(config.errorElementId);
      logDiv.innerHTML += `<p>${'Actual Generation data is not found'}</p>`;
    }
    if (combinedGenerationFittedSeries.length === 0){
      const logDiv = document.getElementById(config.errorElementId);
      logDiv.innerHTML += `<p>${'Actual Generation data is not found'}</p>`;
    }

    // --- total load data
    let combinedLoadActualSeries = [];
    let combinedLoadFittedSeries = [];

    for (const regionCfg of config.regionConfigs) {
      const checkBox = document.getElementById(regionCfg.checkboxId);
      if (!checkBox || !checkBox.checked) continue;

      const series = await getCombinedLoadSeries(
        regionCfg.load_var_label,
        'forecast_prev_actual.json',
        'forecast_prev_fitted.json',
        'forecast_curr_fitted.json',
        config
      );
      if (series) {
        combinedLoadActualSeries.push(series.actualSeries); // Store actual separately
        combinedLoadFittedSeries.push(series.fittedSeries); // Store fitted separately
      }
    }

//    if (!forecastStartTime_) {
//        forecastStartTime_ = forecastStartTime;
//    }

//      const logDiv = document.getElementById(config.errorElementId);
//      logDiv.innerHTML += `<p>${forecastStartTime_}</p>`;


    // --- carbon intensity graph data
    let combinedCarbonActualSeries = [];
    let combinedCarbonFittedSeries = [];

    for (const regionCfg of config.regionConfigs) {
      const checkBox = document.getElementById(regionCfg.checkboxId);
      if (!checkBox || !checkBox.checked) continue;

      const series = await getCombinedLoadSeries(
        regionCfg.carbon_var_label,
        'forecast_prev_actual.json',
        'forecast_prev_fitted.json',
        'forecast_curr_fitted.json',
        config
      );
      if (series) {
        combinedCarbonActualSeries.push(series.actualSeries); // Store actual separately
        combinedCarbonFittedSeries.push(series.fittedSeries); // Store fitted separately
      }
    }

    updateStackedChart(config, finalSeries, forecastStartTime_); // energy mix
    updateTotalLineChart(
        config,
        combinedGenerationActualSeries,
        combinedGenerationFittedSeries,
        combinedLoadActualSeries,
        combinedLoadFittedSeries,
        forecastStartTime_
    ); // load & gen (only fitted)
    updateTotalLineChart2(
        config,
        combinedCarbonActualSeries,
        combinedCarbonFittedSeries,
        forecastStartTime_
    ); // carbon intensity (only fitted)
  };
}


async function getCombinedLoadSeries(
        var_label, pastFileActual, pastForecastFile, currentForecastFile, config) {
  // Fetch data
  const prevActualData = await getCachedData(var_label, pastFileActual, config.errorElementId);
  const prevFittedData = await getCachedData(var_label, pastForecastFile, config.errorElementId);
  const currFittedData = await getCachedData(var_label, currentForecastFile, config.errorElementId);
  let forecastStartTime = null; // Variable to store the forecast start time

  if (!prevFittedData && !currFittedData) return null;

  // Initialize arrays
  const fittedData = [];
  const actualData = [];
  const pastDataRatio = document.getElementById(config.pastDataSliderId)?.value / 100 || 1;

  // Process fitted data
  if (prevFittedData) {
    fittedData.push(...prevFittedData.slice(-Math.floor(prevFittedData.length * pastDataRatio)));
  }
  if (currFittedData) {
    fittedData.push(...currFittedData);
  }

  // Process actual data (only past actuals are available)
  if (prevActualData) {
    actualData.push(...prevActualData.slice(-Math.floor(prevActualData.length * pastDataRatio)));
  }

    // Determine forecast start time by checking the last timestamp in past actual data
    if (Array.isArray(prevActualData) && prevActualData.length > 0) {
      const lastEntry = prevActualData[prevActualData.length - 1];

      if (lastEntry?.x) {
        forecastStartTime = new Date(lastEntry.x).getTime();
      }
    }

    if (!forecastStartTime) {
      const logDiv = document.getElementById(config.errorElementId);
      logDiv.innerHTML += `<p>Error: Unable to determine forecast start time</p>`;
      forecastStartTime = Date.now(); // Set to current timestamp as fallback
    }

//  if (!forecastStartTime){
//      const logDiv = document.getElementById(config.errorElementId);
//      logDiv.innerHTML += `<p>${"Errro"}</p>`;
//      forecastStartTime = 1000;
//  }

//      const logDiv = document.getElementById(config.errorElementId);
//      logDiv.innerHTML += `<p>${"Errro"}</p>`;
//      forecastStartTime = 1000;

  return {
    fittedSeries: {
      name: `${var_label} - Fitted`,
      data: fittedData,
      color: '#008080',
      forecastStartTime: forecastStartTime
    },
    actualSeries: {
      name: `${var_label} - Actual`,
      data: actualData,
      color: '#FF5733',
      forecastStartTime:forecastStartTime
    }
  };
}
//async function getCombinedLoadSeries(var_label, pastFileActual,
//    pastForecastFile, currentForecastFile, config) {
//
//  const prevActualData = await getCachedData(var_label, pastForecastFile, config.errorElementId);
//  const prevFittedData = await getCachedData(var_label, pastForecastFile, config.errorElementId);
//  const currFittedData = await getCachedData(var_label, currentForecastFile, config.errorElementId);
//
//  if (!prevFittedData && !currFittedData) return null;
//
//  const combinedData = [];
//  const pastDataRatio = document.getElementById(config.pastDataSliderId)?.value / 100 || 1;
//
//  if (prevFittedData) {
//    combinedData.push(...prevFittedData.slice(-Math.floor(prevFittedData.length * pastDataRatio)));
//  }
//  if (currFittedData) {
//    combinedData.push(...currFittedData);
//  }
//
//  return {
//    name: `${var_label}`,
//    data: combinedData,
//    color: '#008080'
//  };
//}

/**
 * Main function that wires up all the event listeners and global functions for a given chart configuration.
 */
function setupStackedChartEvents({
  stackedChartNum,
  descriptionToggleId,
  descriptionContainerId,
  descLoadedKey,
  createdKey,
  instanceKey,
  detailsSelector,
  filePrefix,
  getConfigFunction
}) {
  setupDescriptionToggleEvent({ descriptionToggleId, descriptionContainerId, descLoadedKey, filePrefix });
  defineUpdateChartFunction({ stackedChartNum, getConfigFunction });
  defineResetChartFunction({ stackedChartNum, createdKey, instanceKey });
  setupDetailsToggleEvent({ detailsSelector, stackedChartNum, createdKey, instanceKey });
}

// Wire-up all stacked chart events based on the StackedChartConfigs array
StackedChartConfigs.forEach(cfg => setupStackedChartEvents(cfg));