import os.path

import pandas as pd
import requests, json, time, gc
from user_agent import generate_user_agent
from io import StringIO
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta
from pandas.errors import ParserError

from data_collection_modules.parquet_operations import ParquetOperations
# from .utils import validate_dataframe

from logger import get_logger
logger = get_logger(__name__)

class DataEnergySMARD:

    dw_id = 'G2Vtz'  # Importe gesamt
    dw_id_fr = 'XJFzP'  # Importe Frankreich
    dw_id_ch = 'BjzEn'  # Importe Schweiz
    dw_id_nl = 'mOf7y'  # Importe Niederlande
    dw_id_emix = 'Mzofi'  # Strommix
    dw_source_all = 'https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22week%22,%22moduleIds%22:%5B22004629%5D,%22selectedCategory%22:null,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D,%22region%22:%22DE%22%7D'
    dw_source_france = 'https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22week%22,%22moduleIds%22:%5B22004546,22004404%5D,%22selectedCategory%22:null,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D,%22region%22:%22DE%22%7D'
    dw_source_ch = 'https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22week%22,%22moduleIds%22:%5B22004552,22004410%5D,%22selectedCategory%22:null,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D,%22region%22:%22DE%22%7D'
    dw_source_nl = 'https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22week%22,%22moduleIds%22:%5B22004406,22004548%5D,%22selectedCategory%22:null,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D,%22region%22:%22DE%22%7D'

    # power generation
    REALIZED_POWER_GENERATION = [1001224, 1004066, 1004067, 1004068,
                                 1001223, 1004069, 1004071, 1004070, 1001226, 1001228, 1001227, 1001225]
    INSTALLED_POWER_GENERATION = [3004072, 3004073, 3004074, 3004075,
                                  3004076, 3000186, 3000188, 3000189, 3000194, 3000198, 3000207, 3003792]
    FORECASTED_POWER_GENERATION = [ 2000122, 2000715, 2000125, 2003791, 2000123 ]

    # power consumption
    FORECASTED_POWER_CONSUMPTION = [6000411, 6004362]
    REALIZED_POWER_CONSUMPTION = [5000410]
    REALIZED_POWER_CONSUMPTION_RESIDUAL = [5004359]

    # market
    WHOLESALE_PRICES = [8004169, 8004170, 8000252, 8000253, 8000251, 8000254,
                        8000255, 8000256, 8000257, 8000258, 8000259, 8000260, 8000261, 8000262]
    COMMERCIAL_FOREIGN_TRADE = [
        8004169, 8004170, 8000252, 8000253, 8000251, 8000254,
        8000255, 8000256, 8000257, 8000258, 8000259, 8000260, 8000261, 8000262
    ]
    PHYSICAL_POWER_FLOW = [
        31000714, 31000140, 31000569, 31000145, 31000574, 31000570, 31000139, 31000568,
        31000138, 31000567, 31000146, 31000575, 31000144, 31000573, 31000142, 31000571,
        31000143, 31000572, 3100014
    ]

    # commercial trade Germany/France
    COMMERCIAL_TRADE_FR = [22004546, 22004404]  # first import
    # commercial trade Germany/Netherlands
    COMMERCIAL_TRADE_NL = [22004548, 22004406]
    # commercial trade Germany/Belgium
    COMMERCIAL_TRADE_BE = [22004712, 22004998]
    # commercial trade Germany/Czechia
    COMMERCIAL_TRADE_CZ = [22004553, 22004412]
    # commercial trade Germany/Switzerland
    COMMERCIAL_TRADE_CH = [22004552, 22004410]
    # commercial trade Germany/Poland
    COMMERCIAL_TRADE_PL = [22004550, 22004408]
    # commercial trade Germany/Norway
    COMMERCIAL_TRADE_NO = [22004724, 22004722]
    # commercial trade Germany/Denmark
    COMMERCIAL_TRADE_DK = [22004545, 22004403]
    # commercial trade Germany/Sweden
    COMMERCIAL_TRADE_SC = [22004551, 22004409]
    # commercial trade Germany/Luxemburg
    COMMERCIAL_TRADE_LU = [22004547, 22004405]
    # commercial trade Germany/Austria
    COMMERCIAL_TRADE_AT = [22004549, 22004407]

    # commercial trade all countries
    COMMERCIAL_TRADE_ALL = [22004629]


    country_map = {
        'france':COMMERCIAL_TRADE_FR,
        'belgium':COMMERCIAL_TRADE_BE,
        'switzerland':COMMERCIAL_TRADE_CH,
        'czechia':COMMERCIAL_TRADE_CZ,
        'denmark':COMMERCIAL_TRADE_DK,
        'netherlands':COMMERCIAL_TRADE_NL,
        'norway':COMMERCIAL_TRADE_NO,
        'poland':COMMERCIAL_TRADE_PL,
        'sweden':COMMERCIAL_TRADE_SC,
        'luxembourg':COMMERCIAL_TRADE_LU,
        'austria':COMMERCIAL_TRADE_AT
    }

    # spot market
    SPOT_MARKET = [8004169]

    mapping_energy = {
        'Datum':'date',
        'Biomasse [MWh] Originalauflösungen' : "biomass",
        'Kernenergie [MWh] Originalauflösungen': 'nuclear_energy',
        'Erdgas [MWh] Originalauflösungen': 'natural_gas',
        'Pumpspeicher [MWh] Originalauflösungen': 'pumped_storage',
        'Sonstige Konventionelle [MWh] Originalauflösungen' : 'other_conventional',
        'Braunkohle [MWh] Originalauflösungen' : 'lignite',
        'Steinkohle [MWh] Originalauflösungen' : 'hard_coal',
        'Sonstige Erneuerbare [MWh] Originalauflösungen' : 'other_renewables',
        'Wasserkraft [MWh] Originalauflösungen' : 'hydropower',
        'Wind Offshore [MWh] Originalauflösungen' : 'wind_offshore',
        'Wind Onshore [MWh] Originalauflösungen' : 'wind_onshore',
        'Photovoltaik [MWh] Originalauflösungen' : 'solar',
        'Sonstige [MWh] Berechnete Auflösungen' : 'other',
        'Sonstige [MWh] Originalauflösungen' : 'other'
    }
    mapping_load = {
        'Gesamt (Netzlast) [MWh] Originalauflösungen':'total_grid_load',
        'Gesamt [MWh] Berechnete Auflösungen':'total_load',
        'Gesamt [MWh] Originalauflösungen':'total',
        'Residuallast [MWh] Originalauflösungen':'residual_load',
        'Netzlast [MWh] Originalauflösungen':'total_grid_load' # new column name...
    }
    mapping_prices = {
        'Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen':'spot_price',
        'Deutschland/Luxemburg [€/MWh] Originalauflösungen':'spot_price'
    }
    mapping_cross_border = {
        'Nettoexport [MWh] Originalauflösungen' : 'net_export',
        'Frankreich (Export) [MWh] Originalauflösungen':'france_export',
        'Frankreich (Import) [MWh] Originalauflösungen':'france_import',
        'Belgien (Export) [MWh] Originalauflösungen': 'belgium_export',
        'Belgien (Import) [MWh] Originalauflösungen': 'belgium_import',
        'Schweiz (Export) [MWh] Originalauflösungen':'switzerland_export',
        'Schweiz (Import) [MWh] Originalauflösungen':'switzerland_import',
        'Tschechien (Export) [MWh] Originalauflösungen':'czechia_export',
        'Tschechien (Import) [MWh] Originalauflösungen':'czechia_import',
        'Dänemark (Export) [MWh] Originalauflösungen':'denmark_export',
        'Dänemark (Import) [MWh] Originalauflösungen':'denmark_import',
        'Niederlande (Export) [MWh] Originalauflösungen':'netherlands_export',
        'Niederlande (Import) [MWh] Originalauflösungen':'netherlands_import',
        'Norwegen (Export) [MWh] Originalauflösungen':'norway_export',
        'Norwegen (Import) [MWh] Originalauflösungen':'norway_import',
        'Polen (Export) [MWh] Originalauflösungen':'poland_export',
        'Polen (Import) [MWh] Originalauflösungen':'poland_import',
        'Schweden (Export) [MWh] Originalauflösungen':'sweden_export',
        'Schweden (Import) [MWh] Originalauflösungen':'sweden_import',
        'Luxemburg (Export) [MWh] Originalauflösungen':'luxembourg_export',
        'Luxemburg (Import) [MWh] Originalauflösungen':'luxembourg_import',
        'Österreich (Export) [MWh] Originalauflösungen':'austria_export',
        'Österreich (Import) [MWh] Originalauflösungen':'austria_import'
    }
    mapping_wholesale = {
        'Dänemark 1 [€/MWh] Originalauflösungen' : 'denmark_1',
        'Dänemark 2 [€/MWh] Originalauflösungen' : 'denmark_2',
        'Frankreich [€/MWh] Originalauflösungen' : 'france',
        'Niederlande [€/MWh] Originalauflösungen': 'netherlands',
        'Österreich [€/MWh] Originalauflösungen': 'austria',
        'Polen [€/MWh] Originalauflösungen': 'poland',
        'Schweden 4 [€/MWh] Originalauflösungen': 'sweden_4',
        'Schweiz [€/MWh] Originalauflösungen': 'switzerland',
        'Tschechien [€/MWh] Originalauflösungen': 'czechia',
        'DE/AT/LU [€/MWh] Originalauflösungen': 'de_at_lu',
        'Italien (Nord) [€/MWh] Originalauflösungen' : 'italien_nord',
        'Slowenien [€/MWh] Originalauflösungen' : 'slovenia',
        'Ungarn [€/MWh] Originalauflösungen': 'hungary'
    }

    mapping = mapping_energy | mapping_load | mapping_prices | mapping_cross_border | mapping_wholesale

    def __init__(self, start_date:pd.Timestamp, end_date:pd.Timestamp, verbose:bool):
        self.start_date = start_date#-timedelta(milliseconds=1)
        self.end_date = end_date#+timedelta(milliseconds=1)
        self.verbose=verbose

    @staticmethod
    def convert_to_float(value):
        if value == '' or pd.isna(value):  # Check if the value is an empty string or NaN
            return None  # Return None or some appropriate value for your context
        value = value.replace('.', '')  # Remove thousands separator
        value = value.replace(',', '.')  # Replace decimal separator
        try:
            return float(value)
        except ValueError:
            return None  # Handle other cases where conversion is not possible

    @staticmethod
    def requestSmardData(  # request smard data with default values
            modulIDs=[8004169],
            timestamp_from_in_milliseconds=(int(time.time()) * 1000) - (3*3600)*1000,
            timestamp_to_in_milliseconds=(int(time.time()) * 1000),
            region="DE",
            language="de",
            type="discrete",
            verbose:bool=True
    ):

        s = requests.Session()
        retries = Retry(total=10, backoff_factor=1, status_forcelist=[502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))


        # http request content
        headers = generate_user_agent()
        url = "https://www.smard.de/nip-download-manager/nip/download/market-data"
        body = json.dumps({
            "request_form": [
                {
                    "format": "CSV",
                    "moduleIds": modulIDs,
                    "region": region,
                    "timestamp_from": int(timestamp_from_in_milliseconds),
                    "timestamp_to": int(timestamp_to_in_milliseconds),
                    "type": type,
                    "language": language,
                    # "resolution":"original"#"quarterhour",
                }]})

        # http response
        data = s.post(url, body, headers={
            'user-agent': headers, 'Cache-Control': 'no-cache', 'Pragma': 'no-cache', 'Content-Type': 'application/json'

        })
        if verbose:
            logger.info(f"\tStatus Code: {int(data.status_code)} "
                        f"for modulID: {modulIDs} region: {region} from {int(timestamp_from_in_milliseconds)} "
                        f"to {int(timestamp_to_in_milliseconds)} type: {type} language: {language}")

        # create pandas dataframe out of response string (csv)
        df = pd.read_csv(StringIO(data.text), sep=';')

        # convert rows with numbers to float (with wrong decimal)
        cols = df.filter(regex='.*\[MWh]$').columns
        df[cols] = df[cols].replace('-', '')

        return df

    def _requestSmardDataForTimes(self, start_date, end_date, modules, utc:bool=False):

        time.sleep(1)
        df = self.requestSmardData(
            modulIDs=modules,
            timestamp_from_in_milliseconds=int(start_date.timestamp()*1000),
            timestamp_to_in_milliseconds=int(end_date.timestamp()*1000),
            verbose=self.verbose
        )
        # check if data is corrupted
        errors = 0
        while ('Datum bis' not in df.columns) and (errors < 3):
            time.sleep(4)
            errors += 1
            # df = smard.requestSmardData(modulIDs=modules, timestamp_from_in_milliseconds=1625954400000)  # int(time.time()) * 1000) - (24*3600)*373000  = 1 year + last week
            df = self.requestSmardData(
                modulIDs=modules,
                timestamp_from_in_milliseconds=int(start_date.timestamp()*1000),
                timestamp_to_in_milliseconds=int(end_date.timestamp()*1000),
                verbose=self.verbose
            )

        # Process collected data
        if ('Datum bis' in df.columns):
            # fix wrong decimal
            df = df.replace('-', '', regex=False)
            df = df.rename(columns={'Datum von': 'Datum'})
            df.drop('Datum bis', axis=1, inplace=True)
            # convert to floats
            for key in df.keys():
                if not key in ['Datum']:
                    df[key] = df[key].apply(self.convert_to_float)
            # apply mapping
            df.rename(columns=self.mapping, inplace=True)
            if self.verbose:
                logger.info(f"API call successful. Collected df={df.shape}")
            # convert time to UTC
            if utc:
                df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
                df['datetime_utc'] = (df['date']
                                      .dt.tz_localize('Europe/Berlin', ambiguous='infer')
                                      .dt.tz_convert('UTC'))
                df['date'] = df['datetime_utc']
                df.drop('datetime_utc', axis=1, inplace=True)
                df.set_index('date', inplace=True)
                return df

            df.set_index('date', inplace=True)
            return df

        raise ConnectionError("SMARD API call has failed for " +
                              f"\tSMARD api request for {modules} data for "
                              f"{start_date} ({int(start_date.timestamp()*1000)}) to "
                              f"{end_date} ({int(end_date.timestamp()*1000)})")

    def requestSmardDataForTimes(self, modules, utc:bool=False):
        start_date = self.start_date
        end_date = self.end_date
        for i in range(5):
            try:
                result = self._requestSmardDataForTimes(start_date, end_date, modules, utc)
            except Exception as e:
                start_date = start_date - timedelta(days=7)
                if self.verbose:
                    logger.error(f"Attempt {i}/{5}. Parse error in getting modules {modules} Error:\n{e}. "
                      f"Setting earlier start_date by 7 day to {start_date}")
                continue

            return result
        raise ConnectionError(f"API call has failed for {5} attempts")

    def request_data(self, modules_id:list, utc:bool=True):
        return self.requestSmardDataForTimes( modules=modules_id, utc=utc )

    def _check_freq(self, df:pd.DataFrame, freq:str, place:str, type_:str='sum')->pd.DataFrame:

        if freq=='hourly':
            if type_=='sum':
                df = df.resample('h').sum()
            elif type_=='mean':
                df = df.resample('h').mean()
            else:
                raise NotImplementedError(f"Aggregation type {type_} not implemented")

        elif freq == 'minutely_15':
            pass # assuming 15 min is the default data frequency

        else:
            raise NotImplementedError(f"Frequency {freq} not implemented. "
                                        f"Available frequencies: 'hourly', 'minutely_15'")

        # df.reset_index(names=['date'], inplace=True)
        # Ensure the index is sorted
        df.sort_index(inplace=True)
        # Compute the time difference between consecutive timestamps
        time_diffs = df.index.to_series().diff().dropna()
        if (freq == 'minutely_15'):

            # Check if all differences are exactly 15 minutes
            if not (time_diffs == pd.Timedelta(minutes=15)).all():

                # for DA price, if it is not 15 min -- resample using forward-fill
                if type_ == 'mean' and 'spot_price' in df.columns.tolist():
                    df = df.resample('15min').ffill()
                else:
                    raise ValueError(
                        f"Dataframe in {place} contains irregular time intervals:\n{time_diffs.value_counts()}"
                    )

        return df

    def get_international_flow(self, freq:str)->pd.DataFrame:
        if self.verbose: logger.info(f"Collecting cross-border flows for {self.start_date} to {self.end_date} "
                               f"for {list(self.country_map.keys())}")
        df = pd.DataFrame()
        for country in self.country_map.keys():
            df_country = self.request_data(modules_id=DataEnergySMARD.country_map[country])
            if df.empty: df.index = df_country.index

            # create total flow (note Import is always Negative, export is always positive)
            df[f'{country}_export'] = df_country[f'{country}_export'].fillna(0)
            df[f'{country}_import'] = df_country[f'{country}_import'].fillna(0)

        df = self._check_freq(df, freq, 'international_flows')

        return df

    def get_forecasted_generation(self, freq:str)->pd.DataFrame:
        if self.verbose: logger.info(f"Collecting forecaster generation for {self.start_date} to {self.end_date}")
        # o_smard = DataEnergySMARD(start_date=start_date, end_date=end_date)
        df = self.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_GENERATION)
        df.rename(columns={'total':'total_gen'}, inplace=True)
        df.rename(columns={'other':'other_gen'}, inplace=True)
        df = self._check_freq(df, freq, 'forecasted_generation')
        return df

    def get_forecasted_consumption(self,freq:str)->pd.DataFrame:
        if self.verbose: logger.info(f"Collecting forecaster consumption for {self.start_date} to {self.end_date}")
        df = self.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_CONSUMPTION)
        df = self._check_freq(df, freq, 'forecasted_consumption')
        return df

def collect_smard_from_api(start_date:pd.Timestamp, end_date:pd.Timestamp, datadir:str, freq:str, verbose:bool):

    if verbose: logger.info(f"Updating SMARD data from {start_date} to {end_date} for freq: {freq} ")
    o_smard = DataEnergySMARD( start_date=start_date,  end_date=end_date, verbose=verbose)

    # collect cross-border flows
    fname0 = datadir+f'/tmp_smard_flow_{freq}.parquet'
    if os.path.isfile(fname0):
        df_smard_flow = ParquetOperations.read(fname0)
        if verbose: logger.info(f"Loading file {fname0} for freq: {freq} ")
    else:
        df_smard_flow = o_smard.get_international_flow(freq)
        ParquetOperations.save(df_smard_flow, fname0)
        if verbose: logger.info(f"Saving file {fname0} for freq: {freq} ")


    # collect forecasted generation and load
    fname1 = datadir+f'/tmp_smard_gen_forecasted_{freq}.parquet'
    if os.path.isfile(fname1):
        df_smard_gen_forecasted = ParquetOperations.read(fname1)
        if verbose: logger.info(f"Loading file {fname1} for freq: {freq} ")
    else:
        df_smard_gen_forecasted:pd.DataFrame = o_smard.get_forecasted_generation(freq)
        df_smard_gen_forecasted = df_smard_gen_forecasted.rename(
            columns={col: col + "_forecasted" for col in df_smard_gen_forecasted.columns if col != 'date'}
        )
        # df_smard_gen_forecasted = df_smard_gen_forecasted.resample('h', on='date').sum()
        ParquetOperations.save(df_smard_gen_forecasted, fname1)
        if verbose: logger.info(f"Saving file {fname1} for freq: {freq} ")

    # collecting forecasted consumption
    fname2 = datadir+f'/tmp_smard_con_forecasted_{freq}.parquet'
    if os.path.isfile(fname2):
        df_smard_con_forecasted = ParquetOperations.read(fname2)
        if verbose: logger.info(f"Loading file {fname2} for freq: {freq} ")
    else:
        if verbose: logger.info(
            f"Collecting forecasted power consumption for {start_date} to {end_date} for freq: {freq} "
        )
        df_smard_con_forecasted = o_smard.get_forecasted_consumption(freq)
        df_smard_con_forecasted = df_smard_con_forecasted.rename(
            columns={col: col + "_forecasted" for col in df_smard_con_forecasted.columns if col != 'date'}
        )
        # df_smard_con_forecasted = df_smard_con_forecasted.resample('h', on='date').sum()
        ParquetOperations.save(df_smard_con_forecasted, fname2)
        if verbose: logger.info(f"Saving file {fname2} for freq: {freq} ")



    # collect actual realized generation and load
    fname3 = datadir+f'/tmp_smard_gen_realized_{freq}.parquet'
    if os.path.isfile(fname3):
        df_smard_gen_realized = ParquetOperations.read(fname3)
        if verbose: logger.info(f"Loading file {fname3} for freq: {freq} ")
    else:
        if verbose: logger.info(
            f"Collecting realized power generation for {start_date} to {end_date} for freq: {freq} "
        )
        df_smard_gen_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_GENERATION)
        df_smard_gen_realized = o_smard._check_freq(df_smard_gen_realized, freq, 'realized_generation')
        # df_smard_gen_realized = df_smard_gen_realized.resample('h', on='date').sum()
        ParquetOperations.save(df_smard_gen_realized, fname3)
        if verbose: logger.info(f"Saving file {fname3} for freq: {freq} ")

    # collect realized consumption
    fname4 = datadir+f'/tmp_smard_con_realized_{freq}.parquet'
    if os.path.isfile(fname4):
        df_smard_con_realized = ParquetOperations.read(fname4)
        if verbose: logger.info(f"Loading file {fname4} for freq: {freq} ")
    else:
        if verbose: logger.info(f"Collecting realized power consumption for {start_date} to {end_date} for freq: {freq} ")
        df_smard_con_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION)
        df_smard_con_realized = o_smard._check_freq(df_smard_con_realized, freq, 'realized_consumption')
        # df_smard_con_realized = df_smard_con_realized.resample('h', on='date').sum()
        ParquetOperations.save(df_smard_con_realized, fname4)
        if verbose: logger.info(f"Saving file {fname4} for freq: {freq} ")

    # collect realize consumption residual
    fname5 = datadir+f'/tmp_smard_con_res_realized_{freq}.parquet'
    if os.path.isfile(fname5):
        df_smard_con_res_realized = ParquetOperations.read(fname5)
        if verbose: logger.info(f"Loading file {fname5} for freq: {freq} ")
    else:
        if verbose: logger.info(
            f"Collecting realized power consumption residual for {start_date} to {end_date} for freq: {freq} "
        )
        df_smard_con_res_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION_RESIDUAL)
        # df_smard_con_res_realized = df_smard_con_res_realized.resample('h', on='date').sum()
        df_smard_con_res_realized = o_smard._check_freq(df_smard_con_res_realized, freq, 'realized_consumption_residual')
        ParquetOperations.save(df_smard_con_res_realized, fname5)
        if verbose: logger.info(f"Saving file {fname5} for freq: {freq} ")

    # collect DA prices
    fname6 = datadir+f'/tmp_smard_da_prices_{freq}.parquet'
    if os.path.isfile(fname6):
        df_da_prices = ParquetOperations.read(fname6)
        if verbose: logger.info(f"Loading file {fname6} for freq: {freq} ")
    else:
        if verbose: logger.info(f"Collecting DA prices for {start_date} to {end_date} for freq: {freq} ")
        df_da_prices = o_smard.request_data(modules_id=DataEnergySMARD.SPOT_MARKET)
        # df_da_prices = df_da_prices.resample('h', on='date').mean()
        df_da_prices = o_smard._check_freq(df_da_prices, freq, 'spot_market_price', 'mean')
        ParquetOperations.save(df_da_prices, fname6)
        if verbose: logger.info(f"Saving file {fname6} for freq: {freq} ")


    # merge data
    df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_gen_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_res_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_da_prices,left_index=True,right_index=True,how='outer')

    if verbose: logger.info(f"Deleting temporary files for freq: {freq} ")
    for f in [fname0, fname1, fname2, fname3, fname4, fname5, fname6]:
        if os.path.isfile(f):
            os.remove(f)

    return df_smard

def _update_log(data_dir: str, freq: str, start_date, end_date,
                n_rows_before: int, n_cols_before: int,
                n_rows_after: int, n_cols_after: int,
                n_nans_added: int, size_before_mb: float,
                columns: list[str] | None = None):
    """Append a run entry to log.json (create if missing)."""
    log_path = data_dir + 'log.json'
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    entry = {
        "datetime_utc":   datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "freq":           freq,
        "start_date":     str(start_date),
        "end_date":       str(end_date),
        "n_rows_before":  n_rows_before,
        "n_cols_before":  n_cols_before,
        "n_rows_after":   n_rows_after,
        "n_cols_after":   n_cols_after,
        "n_nans_added":   n_nans_added,
        "size_before_mb": round(size_before_mb, 2),
    }
    if columns is not None:
        entry["columns"] = columns

    log.append(entry)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

def update_smard_from_api(today: pd.Timestamp, data_dir: str, freq: str, verbose: bool):
    """Update SMARD data."""
    if verbose: logger.info(f"Updating SMARD data up to {today}")
    fname = data_dir + f'history_{freq}.parquet'
    df_hist = ParquetOperations.read(fname)
    n_rows_before, n_cols_before = df_hist.shape

    last_timestamp = pd.Timestamp(df_hist.dropna(how='all').last_valid_index())
    start_date_ = last_timestamp - timedelta(hours=72)
    end_date_ = today + timedelta(hours=24)
    df_smard = collect_smard_from_api(
        start_date=start_date_, end_date=end_date_, datadir=data_dir, freq=freq, verbose=verbose
    )

    for col in df_hist.columns:
        if col not in df_smard.columns:
            logger.error(f"Column mismatch between df_hist and df_smard")
            logger.error(f"Expected cols: {df_hist.columns.tolist()}")
            logger.error(f"Actual cols: {df_smard.columns.tolist()}")
            raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue.")

    n_nans_added = int(df_smard.isna().sum().sum())

    if freq == 'hourly':
        df_hist = pd.concat([df_hist[:start_date_ - timedelta(hours=1)], df_smard[start_date_:]], axis=0)
    elif freq == 'minutely_15':
        df_hist = pd.concat([df_hist[:start_date_ - timedelta(minutes=15)], df_smard[start_date_:]], axis=0)
    else:
        raise NotImplementedError(f"freq={freq} not implemented")

    df_hist.sort_index(inplace=True)
    ParquetOperations.save(df_hist, fname)

    _update_log(data_dir, freq, start_date_, end_date_, n_rows_before, n_cols_before, *df_smard.shape, n_nans_added, size_before_mb=ParquetOperations.memory_mb(df_hist))

    if verbose: logger.info(f"SMARD data for freq: {freq} saved to {fname} with shape {df_hist.shape}")
    gc.collect()


def create_smard_from_api(start_date: pd.Timestamp | None, today: pd.Timestamp,
                          data_dir: str, freq: str, verbose: bool):
    """Create SMARD data from API."""
    if verbose: logger.info(f"Collecting SMARD data for {start_date} - {today}")
    fname = data_dir + f'history_{freq}.parquet'
    end_date = today + timedelta(hours=24)
    start_date_ = start_date - timedelta(hours=24)
    df_smard = collect_smard_from_api(
        start_date=start_date_, end_date=end_date, datadir=data_dir, freq=freq, verbose=verbose
    )
    df_smard = df_smard[start_date:today]
    ParquetOperations.save(df_smard, fname)

    _update_log(data_dir, freq, start_date_, end_date,
                n_rows_before=0, n_cols_before=0,
                n_rows_after=df_smard.shape[0], n_cols_after=df_smard.shape[1],
                n_nans_added=int(df_smard.isna().sum().sum()),
                columns=df_smard.columns.tolist(),
                size_before_mb=ParquetOperations.memory_mb(df_smard))

    if verbose: logger.info(f"SMARD data for freq: {freq} saved to {fname} with shape {df_smard.shape}")

if __name__ == '__main__':
    today = datetime.today()

    start_date = pd.Timestamp(datetime(year=2024, month=2, day=1), tz='UTC')
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    end_date = today


    create_smard_from_api(start_date, today, '../database_15min/', freq='minutely_15', verbose=True)

    exit(0)
