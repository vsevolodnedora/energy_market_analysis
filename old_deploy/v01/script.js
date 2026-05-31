// ----------------- LANGUAGE --------------------------
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


// Update all elements with [data-i18n] using i18next
function updateContent() {
  document.querySelectorAll('[data-i18n]').forEach(element => {
      const key = element.getAttribute('data-i18n');
      element.innerHTML = i18next.t(key);
  });
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
      const fileName = `wind_offshore_notes_${language}.md`;
      await loadMarkdown(`data/forecasts/${fileName}`, 'chart1-description-container');
  }

  // Update the text of the language toggle button
  const languageToggleButton = document.getElementById('language-toggle');
  languageToggleButton.textContent = (newLang === 'en') ? 'DE' : 'EN'; // Show the other language
}

// Initialize the application
initializeI18n();

  


// ----------------- OTHER --------------------------
  
let baseUrl = "https://raw.githubusercontent.com/vsevolodnedora/energy_market_analysis/main/deploy/";
let isDarkMode = true;

function toggleDarkMode() {
  const body = document.body;
  body.classList.toggle('dark-mode');
  isDarkMode = !isDarkMode;
  
  // If charts exist, refresh them
  if (chartInstance1) updateChart1();
  if (chartInstance2) updateChart2();
}
  


// A helper to track whether each chart was created
let chart1Created = false;
let chart2Created = false;
let chartInstance1 = null;
let chartInstance2 = null;
  
let chart1DescLoaded = false;
let chart2DescLoaded = false;
  
// Common chart options that can be reused
function getBaseChartOptions() {
  return {
      chart: {
          type: 'line',
          height: 350,
          toolbar: { show: true }
      },
      series: [], // Add your series data here
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
                      minute: '2-digit',
                      hour12: false,
                  });
                  return dateFormatter.format(new Date(timestamp));
              }
          },
          title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }
      },
      yaxis: {
          title: {
              text: i18next.t('offshore-power-label'),
              style: { color: isDarkMode ? '#e0e0e0' : '#000' }
          },
          labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } }
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
  

function lightenColor(color, percent) {
  const num = parseInt(color.slice(1), 16),
      amt = Math.round(2.55 * percent),
      R = (num >> 16) + amt,
      G = (num >> 8 & 0x00FF) + amt,
      B = (num & 0x0000FF) + amt;
      return `#${(
              0x1000000 +
              (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x10000 +
              (G < 255 ? (G < 1 ? 0 : G) : 255) * 0x100 +
              (B < 255 ? (B < 1 ? 0 : B) : 255)
          ).toString(16).slice(1).toUpperCase()
      }`;
}

// document.querySelector('details:nth-of-type(2)').addEventListener('toggle', async function(e) {
//     if (e.target.open && !chart2Created) {
//         chart2Created = true;
//         chartInstance2 = await createChart('#chart2');
//         updateChart2(); // first update
//     }
// });


  


// Listen for toggle on the chart #1 description details
document
.getElementById('chart1-description-details')
.addEventListener('toggle', async function (e) {
  // If the user is opening the details and it's not loaded yet...
  if (e.target.open && !chart1DescLoaded) {
      chart1DescLoaded = true;
      
      // Determine the language-specific file
      const language = i18next.language; // Get the current language ('en' or 'de')
      const fileName = `wind_offshore_notes_${language}.md`;
      
      // Load the appropriate Markdown file
      await loadMarkdown(`data/forecasts/${fileName}`, 'chart1-description-container');
  }
});   
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



// Helper color maps, etc.
const colorMap = {
'wind_offshore_50hz': '#1E90FF',
'wind_offshore_tenn': '#FF6347',
'wind_offshore': '#9370DB'
};

const aliases = {
'wind_offshore_50hz': '50Hz',
'wid_offshore_tenn': 'TenneT',
'wind_offshore': 'Total'
};


// Create a new chart in a given container
async function createChart(containerSelector) {
  const options = getBaseChartOptions();
  const newChart = new ApexCharts(document.querySelector(containerSelector), options);
  await newChart.render();
  return newChart;
}

// Listen for the toggle event on each <details> to create Chart #1 or #2
document.querySelector('details:nth-of-type(1)').addEventListener('toggle', async function(e) {
  if (e.target.open && !chart1Created) {
      chart1Created = true;
      chartInstance1 = await createChart('#chart1');
      updateChart1(); // first update
  }
});
  

// ----------------------------------------
async function fetchData(variable, file) {
  try {
      // Attempt to fetch data from the default location
      const response = await fetch(`data/forecasts/${variable}/${file}`);
      if (!response.ok) throw new Error(`Failed to load ${variable} from default location`);
      const data = await response.json();
      return data.map(([timestamp, value]) => ({ x: new Date(timestamp), y: value }));
  } catch (error) {
      console.warn(error.message);
      try {
          // Attempt to fetch data from the baseUrl fallback
          const fallbackResponse = await fetch(`${baseUrl}data/forecasts/${variable}/${file}`);
          if (!fallbackResponse.ok) throw new Error(`Failed to load ${variable} from fallback URL`);
          const fallbackData = await fallbackResponse.json();
          return fallbackData.map(([timestamp, value]) => ({ x: new Date(timestamp), y: value }));
      } catch (fallbackError) {
          // Handle failure from both locations
          document.getElementById('error-message').textContent = fallbackError.message;
          return null;
      }
  }
}

// Example update function for Chart #1
async function updateChart1() {

  // If chart not yet created, do nothing
  if (!chartInstance1) return;

  document.getElementById('error-message').textContent = '';
  const seriesData = [];
  const annotations = [];

  // Grab controls
  const show50hz      = document.getElementById('50hz-checkbox-1').checked;
  const showTenn      = document.getElementById('tenn-checkbox-1').checked;
  const showTotal     = document.getElementById('total-checkbox-1').checked;
  const pastDataRatio = document.getElementById('past-data-slider-1').value / 100;
  const showInterval  = document.getElementById('showci_checkbox-1').checked; // if show credibility interval

  /**
   * Efficiently fetch data in parallel and then add lines/areas to the chart.
   * @param {string} variable - Key to fetch data for (e.g., 'wind_offshore_50hz').
   * @param {string} prevFittedFile - Past forecast fitted values (e.g., 'forecast_prev_fitted.json').
   * @param {string} prevActualFile - Past forecast actual values (e.g., 'forecast_prev_actual.json').
   * @param {string} prevLowerFile - Past forecast lower credibility interval (e.g., 'forecast_prev_lower.json').
   * @param {string} prevUpperFile - Past forecast upper credibility interval (e.g., 'forecast_prev_upper.json').
   * @param {string} currFittedFile - Current forecast fitted values (e.g., 'forecast_curr_fitted.json').
   * @param {string} currLowerFile - Current forecast lower credibility interval (e.g., 'forecast_curr_lower.json').
   * @param {string} currUpperFile - Current forecast upper credibility interval (e.g., 'forecast_curr_upper.json').
   */

  async function addSeries(
    variable,
    prevFittedFile,
    prevActualFile,
    prevLowerFile,
    prevUpperFile,
    currFittedFile,
    currLowerFile,
    currUpperFile
  ) {
      const baseColor = colorMap[variable];
      const alias     = aliases[variable];

      // Fetch in parallel for efficiency
      // Only fetch prev/curr lower/upper if showInterval is checked.
      const [
        pastFittedData,
        pastActualData,
        pastLowerData,
        pastUpperData,
        currentData,
        currentLowerData,
        currentUpperData
      ] = await Promise.all([
        fetchData(variable, prevFittedFile),
        fetchData(variable, prevActualFile),
        showInterval ? fetchData(variable, prevLowerFile) : null,
        showInterval ? fetchData(variable, prevUpperFile) : null,
        fetchData(variable, currFittedFile),
        showInterval ? fetchData(variable, currLowerFile) : null,
        showInterval ? fetchData(variable, currUpperFile) : null
      ]);

      // -------------------- PAST FITTED --------------------
      if (pastFittedData) {
          const pastToShow = Math.floor(pastFittedData.length * pastDataRatio);
          seriesData.push({
              name : `${alias} (${i18next.t('past-fitted-label')})`, 
              data : pastFittedData.slice(-pastToShow),
              color: baseColor,
              type : 'line'
          });
      }

      // -------------------- PAST ACTUAL --------------------
      if (pastActualData) {
          const pastToShow = Math.floor(pastActualData.length * pastDataRatio);
          seriesData.push({
              name     : `${alias} (${i18next.t('past-actual-label')})`,
              data     : pastActualData.slice(-pastToShow),
              color    : lightenColor(baseColor, 40),
              type     : 'line',
              dashStyle: 'Dash'
          });
      }

      // -------------------- CURRENT FORECAST (Line) --------------------
      if (currentData) {
          seriesData.push({
              name : `${alias} (${i18next.t('current-label')})`,
              data : currentData,
              color: baseColor,
              type : 'line'
          });
          // Add an annotation to mark the first forecast point
          if (currentData.length > 0) {
              annotations.push({
                  x: currentData[0].x.getTime(),
                  borderColor: '#808080',
                  label: {
                    text: i18next.t('last-forecast-label'),
                    style: { color: '#FFFFFF', background: '#808080' }
                  }
              });
          }
      }

      // -------------------- PREV FORECAST INTERVAL (Area) --------------------
      // Apply pastDataRatio to the past interval as well
      if (
        showInterval &&
        pastLowerData &&
        pastUpperData &&
        pastLowerData.length === pastUpperData.length
      ) {
          // Only show the same slice for lower & upper
          const pastLength  = Math.floor(pastLowerData.length * pastDataRatio);
          const lowerSlice  = pastLowerData.slice(-pastLength);
          const upperSlice  = pastUpperData.slice(-pastLength);

          // Construct polygon from lower -> reversed upper
          const forecastPolygon = [
            ...lowerSlice.map((pt) => ({ x: pt.x, y: pt.y })),
            ...upperSlice.slice().reverse().map((pt) => ({ x: pt.x, y: pt.y }))
          ];
          if (forecastPolygon.length > 0) {
              seriesData.push({
                  name        : `${alias} (${i18next.t('prev-forecast-interval-label')})`,
                  type        : 'area',
                  data        : forecastPolygon,
                  color       : baseColor,
                  fillOpacity : 0.7,
                  showInLegend: true,
                  fill: {
                      type: 'gradient',
                      gradient: {
                          shade           : 'light',
                          type            : 'vertical',
                          shadeIntensity  : 0.7,
                          gradientToColors: [baseColor],
                          inverseColors   : false,
                          opacityFrom     : 0.2,
                          opacityTo       : 0.5
                      }
                  },
                  stroke: { width: 1 }
              });
          }
      }

      // -------------------- CURRENT FORECAST INTERVAL (Area) --------------------
      if (
        showInterval &&
        currentLowerData &&
        currentUpperData &&
        currentLowerData.length === currentUpperData.length
      ) {
          // Construct polygon from lower -> reversed upper (full current intervals)
          const forecastPolygon = [
            ...currentLowerData.map((pt) => ({ x: pt.x, y: pt.y })),
            ...currentUpperData.slice().reverse().map((pt) => ({ x: pt.x, y: pt.y }))
          ];
          if (forecastPolygon.length > 0) {
              seriesData.push({
                  name        : `${alias} (${i18next.t('forecast-interval-label')})`,
                  type        : 'area',
                  data        : forecastPolygon,
                  color       : baseColor,
                  stroke: { width: 1 }
              });
          }
      }
  }

  // Conditionally add series for each region
  if (show50hz) {
    await addSeries(
      'wind_offshore_50hz',
      'forecast_prev_fitted.json',
      'forecast_prev_actual.json',
      'forecast_prev_lower.json',
      'forecast_prev_upper.json',
      'forecast_curr_fitted.json',
      'forecast_curr_lower.json',
      'forecast_curr_upper.json'
    );
  }

  if (showTenn) {
    await addSeries(
      'wind_offshore_tenn',
      'forecast_prev_fitted.json',
      'forecast_prev_actual.json',
      'forecast_prev_lower.json',
      'forecast_prev_upper.json',
      'forecast_curr_fitted.json',
      'forecast_curr_lower.json',
      'forecast_curr_upper.json'
    );
  }

  if (showTotal) {
    await addSeries(
      'wind_offshore',
      'forecast_prev_fitted.json',
      'forecast_prev_actual.json',
      'forecast_prev_lower.json',
      'forecast_prev_upper.json',
      'forecast_curr_fitted.json',
      'forecast_curr_lower.json',
      'forecast_curr_upper.json'
    );
  }

  // Example: add "Now" line
  const now = new Date();
  annotations.push({
    x: now.getTime(),
    borderColor: '#FF0000',
    label: { 
      text: i18next.t('now-label'),
      style: { color: '#FFF', background: '#FF0000' }
    }
  });

  // Update the chart
  chartInstance1.updateOptions({
    series: seriesData,
    annotations: { xaxis: annotations },
    tooltip: { theme: isDarkMode ? 'dark' : 'light' },
    xaxis: {
      labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } },
      title:  { style: { color: isDarkMode ? '#e0e0e0' : '#000' } },
    },
    yaxis: {
      title: {
        text : i18next.t('offshore-power-label-mw'),
        style: {
          color   : isDarkMode ? '#e0e0e0' : '#000',
          fontSize: '14px'
        }
      },
      labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } }
    },
    legend: {
      labels: { colors: isDarkMode ? '#e0e0e0' : '#000' }
    }
  });
}

