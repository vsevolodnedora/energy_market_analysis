import os.path

import pandas as pd
import requests, json, time, gc
from user_agent import generate_user_agent
from io import StringIO
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta
from pandas.errors import ParserError

# from .utils import validate_dataframe

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

    # map original quantities to more easily readable ones
    mapping = {
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
        'Sonstige [MWh] Originalauflösungen' : 'other',
        # load
        'Gesamt (Netzlast) [MWh] Originalauflösungen':'total_grid_load',
        'Gesamt [MWh] Berechnete Auflösungen':'total_load',
        'Gesamt [MWh] Originalauflösungen':'total',
        'Residuallast [MWh] Originalauflösungen':'residual_load',
        # prices
        'Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen':'spot_price',
        'Deutschland/Luxemburg [€/MWh] Originalauflösungen':'spot_price',
        # trade
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
        'Österreich (Import) [MWh] Originalauflösungen':'austria_import',
        # wholesale trade
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
            print("\tStatus Code:", data.status_code)
        # print("Response Text:", data.text)

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
        pass

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
                print(f"API call successful. Collected df={df.shape}")
            # convert time to UTC
            if utc:
                df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
                df['datetime_utc'] = (df['date']
                                      .dt.tz_localize('Europe/Berlin', ambiguous='infer')
                                      .dt.tz_convert('UTC'))
                df['date'] = df['datetime_utc']
                df.drop('datetime_utc', axis=1, inplace=True)

                return df
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
                    print(f"Attempt {i}/{5}. Parse error in getting modules {modules} Error:\n{e}. "
                      f"Setting earlier start_date by 7 day to {start_date}")
                continue

            return result
        raise ConnectionError(f"API call has failed for {5} attempts")

    def request_data(self, modules_id:list, utc:bool=True):
        return self.requestSmardDataForTimes( modules=modules_id, utc=utc )
    #
    # def request_power_generation(self, utc:bool=True)->pd.DataFrame:
    #     return self.requestSmardDataForTimes( self.REALIZED_POWER_GENERATION, utc=utc )
    #
    # def request_realized_consumption(self, utc:bool=True ):
    #     return self.requestSmardDataForTimes( self.REALIZED_POWER_CONSUMPTION, utc=utc )
    #
    # def request_spot_market(self, utc:bool=True)->pd.DataFrame:
    #     return self.requestSmardDataForTimes( self.SPOT_MARKET, utc=utc )
    #
    # def request_forecasted_generation(self, utc:bool=True)->pd.DataFrame:
    #     return self.requestSmardDataForTimes( self.FORECASTED_POWER_GENERATION, utc=utc )
    #
    # def request_forecasted_consumption(self, utc:bool=True)->pd.DataFrame:
    #     return self.requestSmardDataForTimes( self.FORECASTED_POWER_CONSUMPTION, utc=utc )

    ''' ------------------------------------------------------------- '''

    def get_international_flow(self)->pd.DataFrame:
        if self.verbose: print(f"Collecting cross-border flows for {self.start_date} to {self.end_date} "
                               f"for {list(self.country_map.keys())}")
        df = pd.DataFrame()
        for country in self.country_map.keys():
            df_country = self.request_data(modules_id=DataEnergySMARD.country_map[country])
            if df.empty: df['date'] = df_country['date']
            # create total flow (note Import is always Negative, export is always positive)
            df[f'{country}_export'] = df_country[f'{country}_export'].fillna(0)
            df[f'{country}_import'] = df_country[f'{country}_import'].fillna(0)
            # df[f'{country}_flow'] = (
            #         df_country[f'{country}_export'].fillna(0)
            #         + df_country[f'{country}_import'].fillna(0)
            # )
        df = df.resample('h', on='date').sum()
        df.reset_index(names=['date'], inplace=True)
        return df

    def get_forecasted_generation(self)->pd.DataFrame:
        if self.verbose: print(f"Collecting forecaster generation for {self.start_date} to {self.end_date}")
        # o_smard = DataEnergySMARD(start_date=start_date, end_date=end_date)
        df = self.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_GENERATION)
        df.rename(columns={'total':'total_gen'}, inplace=True)
        df.rename(columns={'other':'other_gen'}, inplace=True)
        df = df.resample('h', on='date').sum()
        df.reset_index(names=['date'], inplace=True)
        return df

    def get_forecasted_consumption(self)->pd.DataFrame:
        if self.verbose: print(f"Collecting forecaster consumption for {self.start_date} to {self.end_date}")
        df = self.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_CONSUMPTION)
        # df.rename(columns={'total':'total_gen'}, inplace=True)
        # df.rename(columns={'other':'other_gen'}, inplace=True)
        df = df.resample('h', on='date').sum()
        df.reset_index(names=['date'], inplace=True)
        return df

    def get_smard_da_prices_from_api(self)->pd.DataFrame:
        # Day-ahead prices from SMARD (date is in ECT)
        if self.verbose: print(f"Collecting DA auction price for {self.start_date} to {self.end_date}")
        df_smard_da = self.request_data(modules_id=DataEnergySMARD.SPOT_MARKET)
        df_smard_da = df_smard_da.resample('h', on='date').mean()
        df_smard_da.reset_index(names=['date'], inplace=True)
        # this data is in UTC initially, it seems, so I need to do conversion
        # df_smard_da['date'] = df_smard_da['date'].dt.tz_localize('UTC')
        # df_smard_da['date'] = df_smard_da['date'].dt.tz_convert('Etc/GMT+1')#('Etc/GMT+1')
        return df_smard_da

    # def request_realized_consumption(self, utc:bool=True)->pd.DataFrame:
    #     modules = self.REALIZED_POWER_CONSUMPTION
    #     df = self.requestSmardData(
    #         modulIDs=modules,
    #         timestamp_from_in_milliseconds=int(self.start_date.timestamp()*1000),
    #         timestamp_to_in_milliseconds=int(self.end_date.timestamp()*1000)
    #     )  # last day of 2022
    #
    #     # check if data is corrupted
    #     errors = 0
    #     while ('Datum bis' not in df.columns) and (errors < 3):
    #         time.sleep(2)
    #         errors += 1
    #     if ('Datum bis' in df.columns):
    #         df = self.requestSmardData(
    #             modulIDs=modules,
    #             timestamp_from_in_milliseconds=int(self.start_date.timestamp()*1000),
    #             timestamp_to_in_milliseconds=int(self.end_date.timestamp()*1000)
    #         )  # last day of 2022
    #         # fix wrong decimal
    #         df = df.replace('-', '', regex=False)
    #         df = df.rename(columns={'Datum von': 'Datum'})
    #         # df.index=pd.to_datetime(df["Datum"],format='%d.%m.%Y %H:%M')#'%d.%m.%Y %H:%M')
    #         df.drop('Datum bis', axis=1, inplace=True)
    #         # df.drop('Datum', axis=1, inplace=True)
    #         # df.dropna(axis='columns', inplace=True)
    #         # df.to_csv('./data/power_consumption.tsv',
    #         #           sep='\t', encoding='utf-8', index=False)
    #         # df = pd.read_csv('./data/power_consumption.tsv', sep='\t', thousands='.',
    #         #                  decimal=',', index_col=None, dtype={'Datum': 'string'})
    #
    #         # convert dates
    #         # df['Datum'] = pd.to_datetime(df['Datum'], format="%d.%m.%Y %H:%M")
    #         for key in df.keys():
    #             if not key in ['Datum']:
    #                 df[key] = df[key].apply(self.convert_to_float)
    #
    #         # df = df.groupby(['Datum']).sum()
    #
    #         # convert to week and drop first and last row with partial values
    #         # df.reset_index(inplace=True)
    #         # df = df.resample('h').sum()
    #         # no drop for step-after chart
    #         # df.drop(df.tail(1).index, inplace=True)
    #         # df.drop(df.head(1).index, inplace=True)
    #
    #         # save tsv
    #         # df.to_csv('./data/power_consumption.tsv', sep='\t',
    #         #            encoding='utf-8', index=True)
    #         df.rename(columns=self.mapping, inplace=True)
    #         if utc:
    #             df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
    #             df['datetime_utc'] = (df['date']
    #                                   .dt.tz_localize('Europe/Berlin', ambiguous='infer')
    #                                   .dt.tz_convert('UTC'))
    #             df['date'] = df['datetime_utc']
    #             df.drop('datetime_utc', axis=1, inplace=True)
    #         return df

    # def request_spot_market(self, utc:bool=True)->pd.DataFrame:
    #     modules = self.SPOT_MARKET
    #     df = self.requestSmardData(
    #         modulIDs=modules,
    #         timestamp_from_in_milliseconds=int(self.start_date.timestamp()*1000),
    #         timestamp_to_in_milliseconds=int(self.end_date.timestamp()*1000)
    #     )  # last day of 2022
    #
    #     # check if data is corrupted
    #     errors = 0
    #     while ('Datum bis' not in df.columns) and (errors < 3):
    #         time.sleep(2)
    #         errors += 1
    #     if ('Datum bis' in df.columns):
    #         df = self.requestSmardData(
    #             modulIDs=modules,
    #             timestamp_from_in_milliseconds=int(self.start_date.timestamp()*1000),
    #             timestamp_to_in_milliseconds=int(self.end_date.timestamp()*1000)
    #         )  # last day of 2022
    #         # fix wrong decimal
    #         df = df.replace('-', '', regex=False)
    #         df.rename(columns={'Datum von': 'Datum'}, inplace=True)
    #         # df.index=pd.to_datetime(df["Datum"],format='%d.%m.%Y %H:%M')#'%d.%m.%Y %H:%M')
    #         df.drop('Datum bis', axis=1, inplace=True)
    #         # df.drop('Datum', axis=1, inplace=True)
    #         # df.dropna(axis='columns', inplace=True)
    #         # df.to_csv('./data/power_consumption.tsv',
    #         #           sep='\t', encoding='utf-8', index=False)
    #         # df = pd.read_csv('./data/power_consumption.tsv', sep='\t', thousands='.',
    #         #                  decimal=',', index_col=None, dtype={'Datum': 'string'})
    #
    #         # convert dates
    #         # df['Datum'] = pd.to_datetime(df['Datum'], format="%d.%m.%Y %H:%M")
    #         for key in df.keys():
    #             if not key in ['Datum']:
    #                 df[key] = df[key].apply(self.convert_to_float)
    #
    #         # df = df.groupby(['Datum']).sum()
    #
    #         # convert to week and drop first and last row with partial values
    #         # df.reset_index(inplace=True)
    #         # df = df.resample('h').sum()
    #         # no drop for step-after chart
    #         # df.drop(df.tail(1).index, inplace=True)
    #         # df.drop(df.head(1).index, inplace=True)
    #
    #         # save tsv
    #         # df.to_csv('./data/power_consumption.tsv', sep='\t',
    #         #            encoding='utf-8', index=True)
    #         df.rename(columns=self.mapping, inplace=True)
    #
    #         if utc:
    #             df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
    #             df['datetime_utc'] = (df['date']
    #                                   .dt.tz_localize('Europe/Berlin', ambiguous='infer')
    #                                   .dt.tz_convert('UTC'))
    #             df['date'] = df['datetime_utc']
    #             df.drop('datetime_utc', axis=1, inplace=True)
    #         # Handle repeated hours by assigning unique identifiers
    #         # df['unique_datetime'] = df['date'].astype(str) + df.groupby('date').cumcount().astype(str)
    #         #
    #         # # Convert unique datetime back to datetime (this won't be necessary if you handle data directly)
    #         # df['unique_datetime'] = pd.to_datetime(df['unique_datetime'], format='%Y-%m-%d %H:%M:%S')
    #         #
    #         # # Assume the data is in CET and convert to UTC
    #         # df['datetime_utc'] = (df['unique_datetime']
    #         #                       .dt.tz_localize('Europe/Berlin', ambiguous='infer')
    #         #                       .dt.tz_convert('UTC'))
    #
    #         # print(df[['date', 'datetime_utc']])
    #
    #         return df

    # def request_forecasted_power_generation(self, utc:bool=True)->pd.DataFrame:

# def update_smard_from_api(today:pd.Timestamp,data_dir:str,verbose):
#
#     fname = data_dir + 'history.parquet'
#
#     if not os.path.isdir(fname)
#
#         df_hist = pd.read_parquet(fname)
#
#     first_timestamp = pd.Timestamp(df_hist.dropna(how='any', inplace=False).first_valid_index())
#     last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
#
#     # ---------- SET UPDATE TIMES ------------
#     start_date = last_timestamp-timedelta(hours=24)
#     end_date = today+timedelta(hours=24)
#
#     # ---------- UPDATE SMARD -------------
#     print(f"Updating SMARD data from {start_date} to {end_date}")
#     o_smard = DataEnergySMARD( start_date=start_date,  end_date=end_date, verbose=verbose  )
#     df_smard_flow = o_smard.get_international_flow()
#     df_smard_gen_forecasted = o_smard.get_forecasted_generation()
#     df_smard_con_forecasted = o_smard.get_forecasted_consumption()
#     df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_on='date',right_on='date',how='outer')
#     df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_on='date',right_on='date',how='outer')
#     df_smard.set_index('date',inplace=True)
#     df_smard = df_smard[start_date:today]
#
#     # check columns
#     for col in df_hist.columns:
#         if not col in df_smard.columns:
#             raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")
#
#     # combine
#     df_hist = df_hist.combine_first(df_smard)
#     # if not validate_dataframe(df_hist, text="Updated smard df_hist"):
#     #     raise ValueError(f"Failed to validate the updated dataframe for {fname}")
#
#     # save
#     df_hist.to_parquet(fname)
#     if verbose:print(f"SMARD data is successfully saved to {fname} with shape {df_hist.shape}")
#
#     gc.collect()


def collect_smard_from_api(start_date:pd.Timestamp, end_date:pd.Timestamp, datadir:str, verbose:bool):
    datadir += 'tmp_smard/'
    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    if verbose: print(f"Updating SMARD data from {start_date} to {end_date}")
    o_smard = DataEnergySMARD( start_date=start_date,  end_date=end_date, verbose=verbose)

    # collect cross-border flows
    fname0 = datadir+'/smard_smard_flow.parquet'
    if os.path.isfile(fname0):
        df_smard_flow = pd.read_parquet(fname0)
        if verbose: print(f"Loading file {fname0}")
    else:
        df_smard_flow = o_smard.get_international_flow()
        df_smard_flow.set_index('date',inplace=True)
        df_smard_flow.to_parquet(fname0)
        if verbose: print(f"Saving file {fname0}")


    # collect forecasted generation and load
    fname1 = datadir+'/smard_gen_forecasted.parquet'
    if os.path.isfile(fname1):
        df_smard_gen_forecasted = pd.read_parquet(fname1)
        if verbose: print(f"Loading file {fname1}")
    else:
        df_smard_gen_forecasted:pd.DataFrame = o_smard.get_forecasted_generation()
        df_smard_gen_forecasted = df_smard_gen_forecasted.rename(
            columns={col: col + "_forecasted" for col in df_smard_gen_forecasted.columns if col != 'date'}
        )
        df_smard_gen_forecasted = df_smard_gen_forecasted.resample('h', on='date').sum()
        df_smard_gen_forecasted.to_parquet(fname1)
        if verbose: print(f"Saving file {fname1}")

    # collecting forecasted consumption
    fname2 = datadir+'/smard_con_forecasted.parquet'
    if os.path.isfile(fname2):
        df_smard_con_forecasted = pd.read_parquet(fname2)
        if verbose: print(f"Loading file {fname2}")
    else:
        if verbose: print(f"Collecting forecasted power consumption for {start_date} to {end_date}")
        df_smard_con_forecasted = o_smard.get_forecasted_consumption()
        df_smard_con_forecasted = df_smard_con_forecasted.rename(
            columns={col: col + "_forecasted" for col in df_smard_con_forecasted.columns if col != 'date'}
        )
        df_smard_con_forecasted = df_smard_con_forecasted.resample('h', on='date').sum()
        df_smard_con_forecasted.to_parquet(fname2)
        if verbose: print(f"Saving file {fname2}")

    # collect actual realized generation and load
    fname3 = datadir+'/smard_gen_realized.parquet'
    if os.path.isfile(fname3):
        df_smard_gen_realized = pd.read_parquet(fname3)
        if verbose: print(f"Loading file {fname3}")
    else:
        if verbose: print(f"Collecting realized power generation for {start_date} to {end_date}")
        df_smard_gen_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_GENERATION)
        df_smard_gen_realized = df_smard_gen_realized.resample('h', on='date').sum()
        df_smard_gen_realized.to_parquet(fname3)
        if verbose: print(f"Saving file {fname3}")

    # collect realized consumption
    fname4 = datadir+'/smard_con_realized.parquet'
    if os.path.isfile(fname4):
        df_smard_con_realized = pd.read_parquet(fname4)
        if verbose: print(f"Loading file {fname4}")
    else:
        if verbose: print(f"Collecting realized power consumption for {start_date} to {end_date}")
        df_smard_con_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION)
        df_smard_con_realized = df_smard_con_realized.resample('h', on='date').sum()
        df_smard_con_realized.to_parquet(fname4)
        if verbose: print(f"Saving file {fname4}")

    # collect realize consumption residual
    fname5 = datadir+'/smard_con_res_realized.parquet'
    if os.path.isfile(fname5):
        df_smard_con_res_realized = pd.read_parquet(fname5)
        if verbose: print(f"Loading file {fname5}")
    else:
        if verbose: print(f"Collecting realized power consumption residual for {start_date} to {end_date}")
        df_smard_con_res_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION_RESIDUAL)
        df_smard_con_res_realized = df_smard_con_res_realized.resample('h', on='date').sum()
        df_smard_con_res_realized.to_parquet(fname5)
        if verbose: print(f"Saving file {fname5}")

    # collect DA prices
    fname6 = datadir+'/smard_da_prices.parquet'
    if os.path.isfile(fname6):
        df_da_prices = pd.read_parquet(fname6)
        if verbose: print(f"Loading file {fname6}")
    else:
        if verbose: print(f"Collecting DA prices for {start_date} to {end_date}")
        df_da_prices = o_smard.request_data(modules_id=DataEnergySMARD.SPOT_MARKET)
        df_da_prices = df_da_prices.resample('h', on='date').mean()
        df_da_prices.to_parquet(fname6)
        if verbose: print(f"Saving file {fname6}")


    # merge data
    df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_gen_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_res_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_da_prices,left_index=True,right_index=True,how='outer')

    if verbose: print(f"Deleting temporary files")
    for f in [fname0, fname1, fname2, fname3, fname4, fname5, fname6]:
        if os.path.isfile(f):
            os.remove(f)

    return df_smard


def update_smard_from_api(today:pd.Timestamp,data_dir:str,verbose:bool):
    if verbose: print(f"Updating SMARD data up to {today}")
    fname = data_dir + 'history.parquet'
    df_hist = pd.read_parquet(fname)
    last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
    start_date_ = last_timestamp - timedelta(hours=24)
    end_date_ = today + timedelta(hours=24)
    df_smard = collect_smard_from_api(start_date=start_date_, end_date=end_date_, datadir=data_dir, verbose=verbose)
    # check columns
    for col in df_hist.columns:
        if not col in df_smard.columns:
            raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")
    combined_df = pd.concat([df_hist[:start_date_], df_smard[start_date_:]])
    result_df = combined_df[~combined_df.index.duplicated(keep='first')]
    df_hist = result_df.sort_index()
    # combine
    # df_hist = df_hist[:last_timestamp].combine_first(df_smard[last_timestamp:today])
    # save
    df_hist.to_parquet(fname)
    if verbose:print(f"SMARD data is successfully saved to {fname} with shape {df_hist.shape}")
    gc.collect()


def create_smard_from_api(start_date:pd.Timestamp or None, today:pd.Timestamp,data_dir:str,verbose:bool):
    if verbose: print(f"Collecting SMARD data for {start_date} - {today}")
    fname = data_dir + 'history.parquet'
    end_date = today + timedelta(hours=24)
    start_date_ = start_date - timedelta(hours=24)
    df_smard = collect_smard_from_api(start_date=start_date_, end_date=end_date, datadir=data_dir, verbose=verbose)
    df_smard = df_smard[start_date:today]
    df_smard.to_parquet(fname)
    if verbose:print(f"SMARD data is successfully saved to {fname} with shape {df_smard.shape}")

# def update_create_smard_from_api(start_date:pd.Timestamp or None, today:pd.Timestamp,data_dir:str,verbose):
#
#     fname = data_dir + 'history.parquet'
#
#     if not os.path.isdir(fname):
#         if start_date is None:
#             raise ValueError("Start date must be provided to create a new dataframe")
#     else:
#         df_hist = pd.read_parquet(fname)
#         first_timestamp = pd.Timestamp(df_hist.dropna(how='any', inplace=False).first_valid_index())
#         last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
#         start_date = last_timestamp - timedelta(hours=24)
#     end_date = today + timedelta(hours=24)
#
#     # ---------- UPDATE SMARD -------------
#     df_smard = collect_smard_from_api(start_date=start_date, end_date=end_date, verbose=verbose)
#
#
#     # check columns
#     for col in df_hist.columns:
#         if not col in df_smard.columns:
#             raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")
#
#     # combine
#     df_hist = df_hist.combine_first(df_smard)
#     # if not validate_dataframe(df_hist, text="Updated smard df_hist"):
#     #     raise ValueError(f"Failed to validate the updated dataframe for {fname}")
#
#     # save
#     df_hist.to_parquet(fname)
#     if verbose:print(f"SMARD data is successfully saved to {fname} with shape {df_hist.shape}")
#
#     gc.collect()

if __name__ == '__main__':
    today = datetime.today()
    # smard = DataEnergySMARD(start_date=pd.Timestamp(today-timedelta(days=20),tz='UTC'),
    #                         end_date=pd.Timestamp(today+timedelta(days=1),tz='UTC'))
    # df = smard.get_international_flow()[['france_export','france_import']]

    # today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    # today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    #
    # smard = DataEnergySMARD(start_date=today-timedelta(days=20),
    #                         end_date=today+timedelta(days=1))
    # df = smard.get_international_flow()[['france_export','france_import']]

    start_date = pd.Timestamp(datetime(year=2024, month=1, day=1), tz='UTC')
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    end_date = today
    # ---------- UPDATE SMARD -------------
    print(f"Updating SMARD data from {start_date} to {end_date}")
    o_smard = DataEnergySMARD( start_date=start_date,  end_date=end_date, verbose=True)

    # collect cross-border flows
    df_smard_flow = o_smard.get_international_flow()
    # df_smard_flow = df_smard_flow.resample('h', on='date').sum()
    df_smard_flow.set_index('date',inplace=True)


    # collect forecasted generation and load
    df_smard_gen_forecasted = o_smard.get_forecasted_generation()
    df_smard_gen_forecasted = df_smard_gen_forecasted.rename(
        columns={col: col + "_forecasted" for col in df_smard_gen_forecasted.columns if col != 'date'}
    )
    df_smard_gen_forecasted = df_smard_gen_forecasted.resample('h', on='date').sum()
    # df_smard_gen_forecasted.set_index('date',inplace=True)

    df_smard_con_forecasted = o_smard.get_forecasted_consumption()
    df_smard_con_forecasted = df_smard_con_forecasted.rename(
        columns={col: col + "_forecasted" for col in df_smard_con_forecasted.columns if col != 'date'}
    )
    df_smard_con_forecasted = df_smard_con_forecasted.resample('h', on='date').sum()
    # df_smard_con_forecasted.set_index('date',inplace=True)

    # collect actual realized generation and load
    df_smard_gen_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_GENERATION)
    df_smard_gen_realized = df_smard_gen_realized.resample('h', on='date').sum()
    # df_smard_gen_realized.set_index('date',inplace=True)

    df_smard_con_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION)
    df_smard_con_realized = df_smard_con_realized.resample('h', on='date').sum()
    # df_smard_con_realized.set_index('date',inplace=True)

    df_smard_con_res_realized = o_smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION_RESIDUAL)
    df_smard_con_res_realized = df_smard_con_res_realized.resample('h', on='date').sum()
    # df_smard_con_res_realized.set_index('date',inplace=True)

    # collect DA prices
    df_da_prices = o_smard.request_data(modules_id=DataEnergySMARD.SPOT_MARKET)
    df_da_prices = df_da_prices.resample('h', on='date').mean()
    # df_da_prices.set_index('date',inplace=True)


    # merge data
    df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_gen_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_res_realized,left_index=True,right_index=True,how='outer')
    df_smard = pd.merge(left=df_smard,right=df_da_prices,left_index=True,right_index=True,how='outer')

    #df_smard.set_index('date',inplace=True)
    df_smard = df_smard[start_date:today]
    df_smard.to_csv('./tmp.csv')

    # for key, val in DataEnergySMARD.country_map.items():
    #     df = smard.request_data(modules_id=val)
    #     df.set_index('date', inplace=True)
    #     df_sum = df.aggregate(func=sum)
    #     print(key, float( df_sum[f"{key}_export"]+df_sum[f"{key}_import"] ) / 1e6, ' TW')
    # 2024-11-06 12:00:00+00:00 (1730894400000) to 2024-11-13 17:00:00+00:00 (1731517200000)
    # smard = DataEnergySMARD(start_date=pd.Timestamp('2024-11-10 16:00:00+00:00',tz='UTC'),
    #                         end_date=pd.Timestamp('2024-11-18 17:00:00+00:00',tz='UTC'),
    #                         verbose=True)
    # # df = smard.get_international_flow()[['poland_export','poland_import']]
    #
    # print(smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_GENERATION).columns)
    # print(smard.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_GENERATION).columns)
    #
    # print(smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION).columns)
    # print(smard.request_data(modules_id=DataEnergySMARD.REALIZED_POWER_CONSUMPTION_RESIDUAL).columns)
    # print(smard.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_CONSUMPTION).columns)
    #
    # print(smard.request_data(modules_id=DataEnergySMARD.WHOLESALE_PRICES).columns)
    # print(smard.request_data(modules_id=DataEnergySMARD.WHOLESALE_PRICES))

    print(df_smard.columns)
    print(df_smard.head())

    pass