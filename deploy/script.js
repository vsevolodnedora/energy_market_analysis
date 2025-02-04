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
 * 0) Create a CACHE
 ************************************************************/
// Global cache to store data once fetched
const forecastDataCache = {};

/**
 * Fetches a data file and returns it as an array of { x: Date, y: number }.
 * Tries default location first, then a fallback. Results are cached.
 */
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
 * MISSALENOUS  Generic data fetch function with fallback logic
 ************************************************************/
async function fetchData(variable, file, errorElementId) {
  const loc_dir = 'data/forecasts' // location of the forecast files
  try {
    // Attempt to fetch data from the default location
    const response = await fetch(`${loc_dir}/${variable}/${file}`);
    if (!response.ok) throw new Error(`Failed to load ${variable} from default location`);
    const data = await response.json();
    return data.map(([timestamp, value]) => ({ x: new Date(timestamp), y: value }));
  } catch (error) {
    console.warn(error.message);
    try {
      // Attempt fallback URL
      const fallbackResponse = await fetch(`${baseUrl}${loc_dir}/${variable}/${file}`);
      if (!fallbackResponse.ok) throw new Error(`Failed to load ${variable} from fallback URL`);
      const fallbackData = await fallbackResponse.json();
      return fallbackData.map(([timestamp, value]) => ({ x: new Date(timestamp), y: value }));
    } catch (fallbackError) {
      // Handle failure from both locations
      document.getElementById(errorElementId).textContent = fallbackError.message;
      return null;
    }
  }
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
 * 4) The generic “updateChart” function
 *    Pass a config object so you can re-use for onshore, solar, etc.
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

    tooltip: { theme: isDarkMode ? 'dark' : 'light' },
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
 * =========================================================
 ************************************************************/

const tsoColorMap = {
  "50Hertz": "#0000FF",  // Blue
  "TenneT": "#008000",   // Green
  "TransnetBW": "#FF0000", // Red
  "Amprion": "#FFFF00",  // Yellow
  "Total": "#800090"     // Purple
};

/************************************************************
 * 5) Setup for the first chart
 ************************************************************/

// Common chart options that can be reused
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
                      // hour: '2-digit',
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