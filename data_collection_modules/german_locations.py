"""
WARNING
information profided here may be inaccurate as open source data for all the installations is not readily available.
Moreover it is often not clear which TSO is responsible for which wind or solar farm. In several cases I had
to make and educated guess.

I had to collect the data from various sources to get a complete list.

Use it at your own risk.
Any verifications or corrections are highly appreciated!


Sources:
Locations and coordiantes: ChatGPT
Roughness length: https://www.wind101.net/wind-height/index.htm
Roughness factor: https://eurocodes-tools.com/en/roughness-factor-crz/
"""

loc_cities = [
    {"name": "Berlin", "type": "city", "suffix":"_ber",
     "lat": 52.520007, "lon": 13.404954,
     "om_api_pars":{"cell_selection":"land"}},
    {"name": "Munchen", "type": "city", "suffix":"_mun",
     "lat": 48.1351, "lon": 11.5820,
     "om_api_pars":{"cell_selection":"land"}},
    {"name": "Stuttgart", "type": "city", "suffix":"_stut",
     "lat": 48.7791, "lon": 9.1801,
     "om_api_pars":{"cell_selection":"land"}},
    {"name": "Frankfurt", "type": "city", "suffix":"_fran",
     "lat": 50.1109, "lon": 8.6821,
     "om_api_pars":{"cell_selection":"land"}},
    {"name": "Hamburg", "type": "city", "suffix":"_ham",
     "lat": 53.5488, "lon": 9.9872,
     "om_api_pars":{"cell_selection":"land"}}
]

loc_onshore_windfarms = [
    # 50Hertz
    {
        "name": "Hüselitz Wind Farm",
        "capacity": 151.8,
        "n_turbines": 46,
        "lat": 52.5347,
        "lon": 11.7321,
        "elevation": 50, # meters; approximate elevation above the sea level)
        "z0": 0.1, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "location": "Lower Saxony",
        "TSO": "50Hertz",
        "suffix":"_won_hueselitz",
        "type": "onshore wind farm",
        "link": "https://www.gem.wiki/H%C3%BCselitz_wind_farm"
    },
    {
        "name": "Werder/Kessin Wind Farm",
        "capacity": 148.05,
        "n_turbines": 32,
        "lat": 53.7270,
        "lon": 13.3362,
        "elevation": 20, # meters; approximate elevation above the sea level)
        "z0": 0.03, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "TSO": "50Hertz",
        "suffix":"_won_werder",
        "type": "onshore wind farm",
        "link":"https://www.google.com/maps/search/Werder%2FKessin+Wind+Farm"
    },
    {
        "name": "Uckermark Enertrag Wind Farm",
        "capacity": 106,
        "n_turbines": 72,
        "location": "Uckermark district of Brandenburg",
        "lat": 53.3784,
        "lon": 13.9491,
        "elevation": 40, # meters; approximate elevation above the sea level)
        "z0": 0.1, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "TSO": "50Hertz",
        "suffix":"_won_uckermark",
        "type": "onshore wind farm",
    },
    {
        "name": "Feldheim Wind Farm",
        "location": "Potsdam-Mittelmark",
        "capacity": 74,
        "n_turbines": 47,
        "lat": 52.1165,
        "lon": 12.5795,
        "elevation": 60, # meters; approximate elevation above the sea level)
        "z0": 0.1, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "TSO": "50Hertz",
        "suffix":"_won_feldheim",
        "type": "onshore wind farm",
    },

    # TenneT
    {
        "name": "Reußenköge Wind Farm",
        "capacity": 303,
        "n_turbines": 90,
        "lat": 54.645,
        "lon": 8.877,
        "elevation": 2, # meters; approximate elevation above the sea level)
        "z0": 0.1, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "location": "Schleswig-Holstein",
        "TSO": "TenneT",
        "suffix":"_won_reussenkoege",
        "type": "onshore wind farm",
    },
    {
        "name": "Jade Wind Park",
        "capacity": 40.7, # MW
        "n_turbines": 16,
        "lat": 53.5923, # approximate latitude
        "lon": 8.1079, # approximate longitude
        "elevation": 2, # meters; approximate elevation above sea level
        "z0": 0.03, # meters; typical roughness length for open flat terrain with grass
        "terrain_category": "II", # as defined in the Eurocode standards
        "location": "Sengwarden, Wilhelmshaven, Lower Saxony, Germany",
        "TSO": "TenneT",
        "suffix": "_won_jade",
        "type": "onshore wind farm"
    },
    {
        "name": "Bürgerwindpark Veer Dörper",
        "location": "Schleswig-Holstein",
        "capacity": 99.95,  # in megawatts (MW)
        "n_turbines": 44,
        "lat": 54.6819,
        "lon": 9.1684,
        "elevation": 2, # meters; approximate elevation above the sea level)
        "z0": 0.01, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "TSO": "TenneT",
        "suffix":"_won_doerper",
        "type": "onshore wind farm",
        "link":"https://veer-doerper.de/"
    },

    # TransnetBW
    {
        "name": "Windpark Hohenlochen",
        "location": "Oberwolfach and Hausach",
        "capacity": 16.8,
        "n_turbines": 4,
        "lat": 48.3248,
        "lon": 8.1882,
        "elevation": 650, # meters; approximate elevation above the sea level)
        "z0": 0.3, # meters; typical roughness length for forests
        "terrain_category": "III", # as defined in the Eurocode standards,
        "TSO": "TransnetBW",
        "suffix":"_won_hohenlochen",
        "type": "onshore wind farm",
        'link': 'https://www.stadtanzeiger-ortenau.de/oberwolfach/c-lokales/feierliche-einweihung-und-taufe_a67585?'
    },
    # TODO add Bürgerwindpark Südliche Ortenau, Windpark Großer Wald,
    {
        "name": "Windpark Harthäuser Wald",
        "location": "Harthäuser Wald",
        "capacity": 54.9,
        "n_turbines": 18,
        "turbine_type": "Enercon E-115",
        "lat": 49.30139,
        "lon": 9.40658,
        "elevation": 350, # meters; approximate elevation above the sea level)
        "z0": 1.2, # meters; typical roughness length for forests
        "terrain_category": "III", # as defined in the Eurocode standards,
        "TSO": "TransnetBW",
        "suffix":"_won_harthaeuser",
        "type": "onshore wind farm",
        'link': 'https://www.zeag-energie.de/energie-zukunft/windkraft/harthaeuser-wald.html'
    },
    {
        "name": "Windpark Straubenhardt",
        "location": "Gemeinde Straubenhardt im Enzkreis, Baden-Württemberg",
        "capacity": 11,
        "n_turbines": 18,
        "turbine_type": "Siemens SWT-3.0-113",
        "lat": 48.8176,
        "lon": 8.5222,
        "elevation": 400, # meters; approximate elevation above the sea level)
        "z0": 1.2, # meters; typical roughness length for forests
        "terrain_category": "III", # as defined in the Eurocode standards,
        "TSO": "TransnetBW",
        "suffix":"_won_straubenhardt",
        "type": "onshore wind farm",
        'link': 'https://www.gegenwind-kraichgau.de/besuch-des-windparks-straubenhardt'
    },

    # Amprion
    {
        "name": "Windpark Hollich",
        "capacity": 48,
        "n_turbines": 15,
        "lat": 52.1788,
        "lon": 7.3812,
        "elevation": 60, # meters; approximate elevation above the sea level)
        "z0": 0.3, # meter (roughness length)
        "terrain_category": "III", # as defined in the Eurocode standards,
        "location": "Steinfurt, North Rhine-Westphalia",
        "TSO": "Amprion",
        "suffix":"_won_hollich",
        "type": "onshore wind farm",
        'link':'https://www.windpark-hollich.de/'
    },
    {
        "name": "Windpark Coesfeld Letter Bruch",
        "capacity": 52.8,
        "n_turbines": 13,
        "lat": 51.8716,
        "lon": 7.1500,
        "elevation": 80, # meters; approximate elevation above the sea level)
        "z0": 0.03, # meter (roughness length)
        "terrain_category": "II", #  meters; typical roughness length for open farmland
        "location": "Coesfeld, North Rhine-Westphalia",
        "TSO": "Amprion",
        "suffix":"_won_coesfeld",
        "type": "onshore wind farm",
    },
    {
        "name": "Windpark A31 Gescher-Reken",
        "capacity": 45,
        "n_turbines": 15,
        "lat": 51.8500,
        "lon": 7.0000,
        "elevation": 50, # meters; approximate elevation above the sea level)
        "z0": 0.05, # meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards,
        "location": "Borken, North Rhine-Westphalia",
        "TSO": "Amprion",
        "suffix":"_won_a31",
        "type": "onshore wind farm",
    }
]

loc_offshore_windfarms = [
    # TenneT
    {
        "name": "EnBW Hohe See",
        "capacity": 497,
        "n_turbines": 71,
        "type_turbines": "Siemens SWT-7.0-154 (7 MW each)",
        "lat": 54.4333,
        "lon": 6.3166,
        "TSO": "TenneT",
        "suffix":"_woff_enbw",
        "type": "offshore wind farm"
    },
    {
        "name": "Borkum Riffgrund 2 Wind Farm",
        "capacity": 450,
        "n_turbines": 56,
        "type_turbines": "MHI Vestas V164-8.0 MW (8 MW each)",
        "lat": 53.9665,
        "lon": 6.5493,
        "TSO": "TenneT",
        "suffix":"_woff_borkum",
        "type": "offshore wind farm"
    },
    {
        "name":"Veja Mate Offshore Wind Farm",
        "capacity":402,
        "n_turbines": 67,
        'type_turbines':'Siemens SWT-6.0-154 (6 MW each)',
        'lat':54.3212,
        'lon':5.8603,
        "TSO":"TenneT",
        "suffix":"_woff_veja",
        "type": "offshore wind farm"
    },
    {
        "name": "BARD Offshore 1 Wind Farm",
        "capacity": 400,
        "n_turbines": 80,
        "type_turbines": "BARD 5.0 (5 MW each)",
        "lat": 54.3583,
        "lon": 5.9750,
        "TSO": "TenneT",
        "suffix":"_woff_bard",
        "type": "offshore wind farm"
    },
    {
        "name": "Global Tech I Offshore Wind Farm",
        "capacity": 400,
        "n_turbines": 80,
        "type_turbines": "Areva Multibrid M5000 (5 MW each)",
        "lat": 54.5290,
        "lon": 6.5840,
        "TSO": "TenneT",
        "suffix":"_woff_global",
        "type": "offshore wind farm"
    },

    # 50Hertz
    {
        "name": "Wikinger Offshore Wind Farm",
        "capacity": 350,
        "n_turbines": 70,
        "type_turbines": "Adwen AD 5-135 (5 MW each)",
        "lat": 54.834,
        "lon": 14.068,
        "TSO": "50Hertz",
        "note": "Operated by Iberdrola Renovables Deutschland GmbH, with a 51% stake; "
                "Energy Infrastructure Partners holds a 49% stake.",
        "suffix":"_woff_wikinger",
        "type": "offshore wind farm"
    },
    {
        "name": "Arkona Wind Farm",
        "capacity": 385,
        "n_turbines": 60,
        "type_turbines": "Siemens SWT-6.0-154 direct-drive (6 MW each)",
        "lat": 54.782804,
        "lon": 14.121,
        "TSO": "50Hertz",
        "note": "Operated by RWE on behalf of partners",
        "suffix":"_woff_arkona",
        "type": "offshore wind farm"
    }
]

# solarfarms = [
#     {"name": "SolarparkWeesow-Willmersdorf ", "type": "solar farm", "suffix":"_solw",
#      "lat": 52.6506, "lon": 13.6866,  "om_api_pars":{"cell_selection":"sea"}}
# ]


loc_solarfarms = [
    # {
    #     "name": "Solarpark Frankfurt",
    #     "capacity":None, # float; in MW
    #     "n_panels":None, # int; number of panels to
    #     "type_panels":None, # str; what panes are used
    #     "elevation": None, # float; meters; approximate elevation above the sea level)
    #     "z0": None, # float; meter (roughness length)
    #     "terrain_category": None, # as defined in the Eurocode standards,
    #     "location": None, # str; city/area
    #     "size":None, # float; hectares
    #     "lat":None, # float; decimal latitude
    #     "lon":None, # float; decimal longitude
    #     "TSO":None, # str; TSO that connects this solar farm to the grid
    #     "suffix":"_sol_frankfurt",
    #     "type": "solar farm",
    # },

    # 50Hertz
    {
        "name": "Witznitz Solar Farm",
        "capacity": 650,  # in MW
        "n_panels": 1100000,  # number of panels
        "type_panels": "JinkoSolar Tiger Neo modules with Topcon Hot 3.0 technology",  # panel type
        "elevation": 150,  # meters; approximate elevation above sea level
        "z0": 0.05,  # meter (roughness length)
        "terrain_category": "II",  # as defined in the Eurocode standards
        "location": "Leipzig",
        "size": 500,  # hectares
        "lat": 51.1727,  # decimal latitude
        "lon": 12.4024,  # decimal longitude
        "TSO": "50Hertz",  # TSO that connects this solar farm to the grid
        "suffix": "_sol_witznitz",
        "type": "solar farm",
    },
    {
        "name": "Weesow-Willmersdorf Solar PV Park",
        "capacity": 187,  # in MW
        "n_panels": 465000,  # number of panels
        "type_panels": "Trina Solar dual-glass PV modules, each with approximately 400 Wp output",  # panel specifications
        "elevation": None,  # meters; approximate elevation above sea level (specific data not found)
        "z0": 0.05,  # meter (roughness length)
        "terrain_category": None,  # as defined in the Eurocode standards (specific data not found)
        "location": "Werneuchen, Brandenburg",  # City, area
        "size": 164,  # hectares
        "lat": 52.6475,  # decimal latitude
        "lon": 13.6901,  # decimal longitude
        "TSO": "50Hertz",  # Transmission System Operator connecting the solar farm to the grid
        "suffix": "_sol_weesow",
        "type": "solar farm",
    },
    {
        "name": "Tramm-Göthen Solar Park",
        "capacity": 172.0,  # float; in MW
        "n_panels": 420000,  # int; number of panels
        "type_panels": None,  # str; type of panels used
        "elevation": None,  # float; meters; approximate elevation above sea level
        "z0": None,  # float; meter (roughness length)
        "terrain_category": None,  # as defined in the Eurocode standards
        "location": "Tramm and Lewitzrand, Mecklenburg-Western Pomerania, Germany",  # str; city/area
        "size": 248.0,  # float; hectares
        "lat": 53.5253,  # float; decimal latitude
        "lon": 11.6569,  # float; decimal longitude
        "TSO": "50Hertzs",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_tramm",
        "type": "solar farm",
    },
    {
        "name": "Meuro Solar Park",
        "capacity": 166.0,  # float; in MW
        "n_panels": 636000,  # int; number of panels
        "type_panels": "Canadian Solar photovoltaic panels",  # str; type of panels used
        "elevation": 100.0,  # float; meters; approximate elevation above sea level
        "z0": 0.03,  # float; meters; roughness length
        "terrain_category": "II",  # as defined in the Eurocode standards
        "location": "Meuro, Brandenburg, Germany",  # str; city/area
        "size": 200.0,  # float; hectares
        "lat": 51.545,  # float; decimal latitude
        "lon": 13.980,  # float; decimal longitude
        "TSO": "50Hertz",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_meuro",
        "type": "solar farm",
    },
    {
        "name": "Gottesgabe Solar Park",
        "capacity": 153.131,  # float; in MW
        "n_panels": 350000,  # int; number of panels
        "type_panels": "crystalline and bifacial modules",  # str; types of panels used
        "elevation": 5,  # float; meters; approximate elevation above sea level
        "z0": 0.03,  # float; meters; roughness length for open flat terrain
        "terrain_category": "II",  # as defined in the Eurocode standards
        "location": "Gottesgabe, Neuhardenberg, Brandenburg, Germany",  # str; city/area
        "size": 124.36,  # float; hectares
        "lat": 52.6413,  # float; decimal latitude
        "lon": 14.1923,  # float; decimal longitude
        "TSO": "50Hertz",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_gottesgabe",
        "type": "solar farm",
    },

    # TransnetBW
    {
        "name": "Ernsthof Solar Park",
        "capacity": 34.4, # MW
        "n_panels": 155480, # number of panels
        "type_panels": "LDK Solar Energy Systems modules",
        "elevation": 300, # meters; approximate elevation above sea level
        "z0": 0.2, #meters; typical roughness length for open terrain
        "terrain_category": "II", # open terrain with few obstacles
        "location": "Wertheim, Baden-Württemberg, Germany",
        "size": 85, # hectares
        "lat": 49.7074, # decimal latitude
        "lon": 9.4746, # decimal longitude
        "TSO": "TransnetBW",
        "suffix": "_sol_ernsthof",
        "type": "solar farm"
    },
    {
        "name": "Solarpark Erbach",
        "capacity": 5.0,  # float; in MW
        "n_panels": 20000,  # int; number of panels
        "type_panels": "Monocrystalline silicon panels",  # str; type of panels used
        "elevation": 400.0,  # float; meters; approximate elevation above sea level
        "z0": 0.1,  # float; meter (roughness length)
        "terrain_category": "Category II",  # str; terrain category as defined in the Eurocode standards
        "location": "Erbach, Baden-Württemberg, Germany",  # str; city/area
        "size": 10.0,  # float; hectares
        "lat": 48.1234,  # float; decimal latitude
        "lon": 9.1234,  # float; decimal longitude
        "TSO": "TransnetBW",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_erbach",
        "type": "solar farm",
    },
    {
        "name": "Solarpark Aalen",
        "capacity": 35.0,  # Estimated capacity in MW
        "n_panels": 100000,  # Estimated number of panels
        "type_panels": "Monocrystalline silicon",
        "elevation": 554.0,  # Average elevation in meters
        "z0": 0.03,  # Estimated roughness length in meters
        "terrain_category": "II",
        "location": "Aalen, Baden-Württemberg, Germany",
        "size": 85.0,  # Estimated size in hectares
        "lat": 48.8187,
        "lon": 10.0691,
        "TSO": "TransnetBW",
        "suffix": "_sol_aalen",
        "type": "solar farm"
    },

    # TenneT
    {
        "name": "Lauingen Energy Park",
        "capacity": 25.7,  # float; in MW
        "n_panels": 306084,  # int; number of panels
        "type_panels": "288,132 thin-film CdTe PV modules by First Solar; 17,952 crystalline silicon panels by Yingli",  # str; types of panels used
        "elevation": 440,  # float; meters; approximate elevation above sea level
        "z0": 0.03,  # float; meters; roughness length
        "terrain_category": "II",  # as defined in the Eurocode standards
        "location": "Lauingen, Bavaria, Germany",  # str; city/area
        "size": 63,  # float; hectares
        "lat": 48.53694,  # float; decimal latitude
        "lon": 10.42417,  # float; decimal longitude
        "TSO": "TenneT",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_lauingen",
        "type": "solar farm"
    },
    {
        "name": "Schlechtenberg Solar Park",
        "capacity": 8.2, # float; in MW
        "n_panels": 33000, # int; number of panels
        "type_panels": "Hanwha SolarOne", # str; what panels are used
        "elevation": 700, # float; meters; approximate elevation above sea level
        "z0": 0.1, # float; meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards
        "location": "Sulzberg, Bavaria",
        "size": 12, # float; hectares
        "lat": 47.68028, # float; decimal latitude
        "lon": 10.39417, # float; decimal longitude
        "TSO": "TenneT TSO GmbH", # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_schlechtenberg",
        "type": "solar farm"
    },
    {
        "name": "Mengkofen Solar Park",
        "capacity": 21.78,  # float; in MW
        "n_panels": 98978,  # int; number of panels
        "type_panels": "Flat-panel PV",  # str; type of panels used
        "elevation": None,  # float; meters; approximate elevation above sea level
        "z0": None,  # float; meter (roughness length)
        "terrain_category": None,  # as defined in the Eurocode standards
        "location": "Mengkofen, Bavaria, Germany",  # str; city/area
        "size": None,  # float; hectares
        "lat": 48.700,  # float; decimal latitude
        "lon": 12.383,  # float; decimal longitude
        "TSO": "TenneT",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_mengkofen",
        "type": "solar farm",
    },
    {
        "name": "Solarpark Eggebek",
        "capacity": 83.6,  # float; in MW
        "n_panels": 360000,  # int; number of panels
        "type_panels": "TSM-PC05 series by Trina Solar",  # str; what panels are used
        "elevation": 20.0,  # float; meters; approximate elevation above sea level
        "z0": 0.03,  # float; meter (roughness length)
        "terrain_category": "II",  # as defined in the Eurocode standards
        "location": "Eggebek, Schleswig-Holstein, Germany",  # str; city/area
        "size": 160.0,  # float; hectares
        "lat": 54.62955,  # float; decimal latitude
        "lon": 9.34596,  # float; decimal longitude
        "TSO": "TenneT",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_eggebek",
        "type": "solar farm",
    },

    # Amprion
    {
        "name": "Solarpark Bottrop",
        "capacity": 10.0,  # in MW
        "n_panels": 40000,  # number of panels
        "type_panels": "Monocrystalline silicon",  # type of panels
        "elevation": 60.0,  # meters above sea level
        "z0": 0.05,  # roughness length in meters
        "terrain_category": "II",  # Eurocode terrain category
        "location": "Bottrop, Germany",
        "size": 20.0,  # in hectares
        "lat": 51.5236,  # decimal latitude
        "lon": 6.9225,  # decimal longitude
        "TSO": "Amprion",  # TSO that connects to the grid
        "suffix": "_sol_bottrop",
        "type": "solar farm",
    },
    {
        "name": "Solarpark Kirchhellen",
        "capacity": 10.0, # float; in MW
        "n_panels": 30000, # int; number of panels
        "type_panels": "monocrystalline or polycrystalline", # str; what panels are used
        "elevation": 50.0, # float; meters; approximate elevation above sea level
        "z0": 0.03, # float; meter (roughness length)
        "terrain_category": "II", # as defined in the Eurocode standards
        "location": "Bottrop, Germany", # str; city/area
        "size": 20.0, # float; hectares
        "lat": 51.5486, # float; decimal latitude
        "lon": 6.9261, # float; decimal longitude
        "TSO": "Amprion", # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_krchhellen",
        "type": "solar farm",
    },
    {
        "name": "Solarpark Nettersheim",
        "capacity": 40.5,  # MW
        "n_panels": 122500,  # Number of panels (assuming 330W per panel)
        "type_panels": "High-efficiency photovoltaic panels",
        "elevation": 300,  # meters
        "z0": 0.03,  # meters
        "terrain_category": "Open terrain with few or no obstacles (Terrain category 0)",
        "location": "Nettersheim, Germany",
        "size": 40.5,  # hectares (assuming 1 hectare per MW)
        "lat": 50.5333,  # decimal latitude
        "lon": 6.5667,  # decimal longitude
        "TSO": "Amprion",
        "suffix": "_sol_nettersheim",
        "type": "solar farm"
    },
    {
        "name": "Solarpark Frankfurt",
        "capacity": 17.4,  # float; in MW
        "n_panels": 17400,  # int; number of panels
        "type_panels": "Vertical bifacial PV modules",  # str; what panels are used
        "elevation": 112,  # float; meters; approximate elevation above the sea level
        "z0": None,  # float; meter (roughness length)
        "terrain_category": None,  # as defined in the Eurocode standards
        "location": "Frankfurt Airport, Frankfurt, Germany",  # str; city/area
        "size": 30.8,  # float; hectares
        "lat": 50.0379,  # float; decimal latitude
        "lon": 8.5622,  # float; decimal longitude
        "TSO": "Amprion GmbH",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_frankfurt",
        "type": "solar farm",
    }

]


all_locations = loc_cities + loc_solarfarms + loc_offshore_windfarms + loc_onshore_windfarms

de_regions = [
    {'name':'DE_AMPRION','suffix':'_ampr', 'TSO':'Amprion'},
    {'name':'DE_50HZ','suffix':'_50hz', 'TSO':'50Hertz'},
    {'name':'DE_TENNET','suffix':'_tenn', 'TSO':'TenneT'},
    {'name':'DE_TRANSNET','suffix':'_tran', 'TSO':'TransnetBW'},
]