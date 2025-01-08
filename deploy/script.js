// GLOBAL DEFINITIONS
let baseUrl = "https://raw.githubusercontent.com/vsevolodnedora/energy_market_analysis/main/deploy/";

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
  
    if (chartInstance1) {
        updateChart1(); // Force chart update to reformat labels/axes
    }
    if (chartInstance2) {
        updateChart2();
    }
  
    // Reload the description in the new language if already loaded
    if (chart1DescLoaded) {
        const language = i18next.language; // Get the new current language
        const fileName = `wind_offshore_notes_${language}.md`  ;
        await loadMarkdown(`data/forecasts/${fileName}`, 'chart1-description-container');
    }
  
    // Update the text of the language toggle button
    const languageToggleButton = document.getElementById('language-toggle');
    languageToggleButton.textContent = (newLang === 'en') ? 'DE' : 'EN'; // Show the other language
}
  

/************************************************************
 * 0) Dark Mode
 ************************************************************/

function toggleDarkMode() {
    const body = document.body;
    body.classList.toggle('dark-mode');
    isDarkMode = !isDarkMode;
    
    // If charts exist, refresh them
    if (chartInstance1) updateChart1();
    if (chartInstance2) updateChart2();
}
let isDarkMode = true;

// A helper to track whether each chart was created
let chart1Created = false;
let chart2Created = false;
let chartInstance1 = null;
let chartInstance2 = null;
    
let chart1DescLoaded = false;
let chart2DescLoaded = false;
    

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
    annotations.push({
      x: currentData[0].x.getTime(),
      label: {
        text: i18next.t('last-forecast-label'),
        style: { color: '#FFFFFF', background: '#808080' }
      }
    });
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
    legend: {
      show: false, // Hides the legend
    },
    // legend: {
    //   labels: { show:false, colors: isDarkMode ? '#e0e0e0' : '#000' },
    // },
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
 * 6) The actual update function for “Offshore” Chart #1
 *    (matching the onChange handlers in the HTML)
 ************************************************************/

//Listen for toggle on the chart #1 description details
document
  .getElementById('description1-toggle-checkbox')
  .addEventListener('click', async function () {
    const content = document.getElementById('chart1-description-container');
    
    // Toggle visibility of the dropdown content
    const isVisible = content.style.display === 'block';
    content.style.display = isVisible ? 'none' : 'block';

    // If opening the dropdown and content is not loaded, load it dynamically
    if (!isVisible && !chart1DescLoaded) {
        chart1DescLoaded = true;
        
        // Determine the language-specific file
        const language = i18next.language; // Get the current language ('en' or 'de')
        const fileName = `wind_offshore_notes_${language}.md`;
        
        // Load the appropriate Markdown file
        await loadMarkdown(`data/forecasts/${fileName}`, 'chart1-description-container');
    }
  }); 

// Toggle listener for the first <details> block
document.querySelector('details:nth-of-type(1)')
  .addEventListener('toggle', async function(e) {
    if (e.target.open && !chart1Created) {
      await initializeI18n();            // loads i18n, sets default language
      chart1Created = true;
      chartInstance1 = await createChart('#chart1', getBaseChartOptions());
      updateChart1(); // do first update
    }
  }); 

async function updateChart1() {
  
  await updateChartGeneric({
    chartInstance   : chartInstance1,
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
  });
}

/************************************************************
 * 6) The actual update function for “Onshore” Chart #2
 *    (matching the onChange handlers in the HTML)
 ************************************************************/

//Listen for toggle on the chart #1 description details
document
  .getElementById('description2-toggle-checkbox')
  .addEventListener('click', async function () {
    const content = document.getElementById('chart2-description-container');
    
    // Toggle visibility of the dropdown content
    const isVisible = content.style.display === 'block';
    content.style.display = isVisible ? 'none' : 'block';

    // If opening the dropdown and content is not loaded, load it dynamically
    if (!isVisible && !chart2DescLoaded) {
        chart2DescLoaded = true;
        
        // Determine the language-specific file
        const language = i18next.language; // Get the current language ('en' or 'de')
        const fileName = `wind_onshore_notes_${language}.md`;
        
        // Load the appropriate Markdown file
        await loadMarkdown(`data/forecasts/${fileName}`, 'chart2-description-container');
    }
  }); 

// Toggle listener for the second <details> block
document.querySelector('details:nth-of-type(1)')
  .addEventListener('toggle', async function(e) {
    if (e.target.open && !chart2Created) {
      await initializeI18n();            // loads i18n, sets default language
      chart2Created = true;
      chartInstance2 = await createChart('#chart2', getBaseChartOptions());
      updateChart2(); // do first update
    }
  });
  
async function updateChart2() {
  
  await updateChartGeneric({
    chartInstance   : chartInstance2,
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
  });
}
