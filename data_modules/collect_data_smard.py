import pandas as pd
import requests, json, time
from user_agent import generate_user_agent
from io import StringIO
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta
from pandas.errors import ParserError

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
        'poland':COMMERCIAL_TRADE_PL
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
        'Residuallast [MWh] Originalauflösungen':'residual_load_forecast',
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
    }

    def __init__(self, start_date:pd.Timestamp, end_date:pd.Timestamp):
        self.start_date = start_date
        self.end_date = end_date

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
            type="discrete"
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
                    "timestamp_from": timestamp_from_in_milliseconds,
                    "timestamp_to": timestamp_to_in_milliseconds,
                    "type": type,
                    "language": language,
                    # "resolution":"original"#"quarterhour",
                }]})

        # http response
        data = s.post(url, body, headers={
            'user-agent': headers, 'Cache-Control': 'no-cache', 'Pragma': 'no-cache'
        })

        # create pandas dataframe out of response string (csv)
        df = pd.read_csv(StringIO(data.text), sep=';')

        # convert rows with numbers to float (with wrong decimal)
        cols = df.filter(regex='.*\[MWh]$').columns
        df[cols] = df[cols].replace('-', '')

        return df

    def requestSmardDataForTimes(self, modules, utc:bool=False):
        for i in range(5):
            try:
                # print(f"Requesting data for {self.start_date} to {self.end_date}")
                df = self.requestSmardData(
                    modulIDs=modules,
                    timestamp_from_in_milliseconds=int(self.start_date.timestamp()*1000),
                    timestamp_to_in_milliseconds=int(self.end_date.timestamp()*1000)
                )
                # check if data is corrupted
                errors = 0
                while ('Datum bis' not in df.columns) and (errors < 3):
                    time.sleep(2)
                    errors += 1
                    # df = smard.requestSmardData(modulIDs=modules, timestamp_from_in_milliseconds=1625954400000)  # int(time.time()) * 1000) - (24*3600)*373000  = 1 year + last week
                    df = self.requestSmardData(
                        modulIDs=modules,
                        timestamp_from_in_milliseconds=int(self.start_date.timestamp()*1000),
                        timestamp_to_in_milliseconds=int(self.end_date.timestamp()*1000)
                    )
                # process successfull
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
                    # convert time to UTC
                    if utc:
                        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
                        df['datetime_utc'] = (df['date']
                                              .dt.tz_localize('Europe/Berlin', ambiguous='infer')
                                              .dt.tz_convert('UTC'))
                        df['date'] = df['datetime_utc']
                        df.drop('datetime_utc', axis=1, inplace=True)

                        return df

                raise Exception("API call has failed")
            except ParserError as e:
                print(f"Attempt {i}/{5}. Parse error in getting modules {modules} Error:{e}")
        raise Exception(f"API call has failed after {5} attempts")

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
        # o_smard = DataEnergySMARD(start_date=start_date, end_date=end_date)
        df = pd.DataFrame()
        for country in ['france','norway','switzerland','denmark','czechia','poland','belgium','netherlands']:
            df_country = self.request_data(modules_id=DataEnergySMARD.country_map[country])
            if df.empty: df['date'] = df_country['date']
            # create total flow (note Import is always Negative, export is always positive
            df[f'{country}_flow'] = df_country[f'{country}_export'].fillna(0) + df_country[f'{country}_import'].fillna(0)
        df = df.resample('h', on='date').mean()
        df.reset_index(names=['date'], inplace=True)
        return df

    def get_forecasted_generation(self)->pd.DataFrame:
        # o_smard = DataEnergySMARD(start_date=start_date, end_date=end_date)
        df = self.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_GENERATION)
        df.rename(columns={'total':'total_gen'}, inplace=True)
        df.rename(columns={'other':'other_gen'}, inplace=True)
        df = df.resample('h', on='date').mean()
        df.reset_index(names=['date'], inplace=True)
        return df

    def get_forecasted_consumption(self)->pd.DataFrame:

        df = self.request_data(modules_id=DataEnergySMARD.FORECASTED_POWER_CONSUMPTION)
        # df.rename(columns={'total':'total_gen'}, inplace=True)
        # df.rename(columns={'other':'other_gen'}, inplace=True)
        df = df.resample('h', on='date').mean()
        df.reset_index(names=['date'], inplace=True)
        return df

    def get_smard_da_prices_from_api(self)->pd.DataFrame:
        # Day-ahead prices from SMARD (date is in ECT)

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


if __name__ == '__main__':
    # todo add tests
    pass