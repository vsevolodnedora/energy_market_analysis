"""
WARNING
information profided here may be inaccurate as open source data for all the installations is not readily available.
I had to collect the data from various sources to get a complete list.
Useit at your own risk.
Any verifications or corrections are highly appreciated.
"""

cities = [
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

onshore_windfarms = [
    # 50Hertz
    {
        "name": "Hüselitz Wind Farm",
        "capacity": 151.8,
        "n_turbines": 46,
        "lat": 52.5347,
        "lon": 11.7321,
        "location": "Lower Saxony",
        "TSO": "50Hertz",
        "suffix":"_won_hueselitz",
        "type": "onshore wind farm",
    },
    {
        "name": "Werder/Kessin Wind Farm",
        "capacity": 148.05,
        "n_turbines": 32,
        "lat": 53.7270,
        "lon": 13.3362,
        "TSO": "50Hertz",
        "suffix":"_won_werder",
        "type": "onshore wind farm",
    },
    {
        "name": "Uckermark Enertrag Wind Farm",
        "capacity": 106,
        "n_turbines": 72,
        "location": "Uckermark district of Brandenburg",
        "lat": 53.3784,
        "lon": 13.9491,
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
        "location": "Schleswig-Holstein",
        "TSO": "TenneT",
        "suffix":"_won_reussenkoege",
        "type": "onshore wind farm",
    },
    {
        "name": "Bürgerwindpark Veer Dörper",
        "location": "Schleswig-Holstein",
        "capacity": 99.95,  # in megawatts (MW)
        "n_turbines": 44,
        "lat": 54.6819,
        "lon": 9.1684,
        "TSO": "TenneT",
        "suffix":"_won_doerper",
        "type": "onshore wind farm",
    },

    # TransnetBW
    {
        "name": "Windpark Hohenlochen",
        "location": "Oberwolfach and Hausach",
        "capacity": 16.8,
        "n_turbines": 4,
        "lat": 48.3248,
        "lon": 8.1882,
        "TSO": "TransnetBW",
        "suffix":"_won_hohenlochen",
        "type": "onshore wind farm",
    },


    # Amprion
    {
        "name": "Windpark Hollich",
        "capacity": 48,
        "n_turbines": 15,
        "lat": 52.1788,
        "lon": 7.3812,
        "location": "Steinfurt, North Rhine-Westphalia",
        "TSO": "Amprion",
        "suffix":"_won_hollich",
        "type": "onshore wind farm",
    },
    {
        "name": "Windpark Coesfeld Letter Bruch",
        "capacity": 52.8,
        "n_turbines": 13,
        "lat": 51.8716,
        "lon": 7.1500,
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
        "location": "Borken, North Rhine-Westphalia",
        "TSO": "Amprion",
        "suffix":"_won_a31",
        "type": "onshore wind farm",
    }
]

offshore_windfarms = [
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

solarfarms = [
    {"name": "SolarparkWeesow-Willmersdorf ", "type": "solar farm", "suffix":"_solw",
     "lat": 52.6506, "lon": 13.6866,  "om_api_pars":{"cell_selection":"sea"}}
]

locations = cities + onshore_windfarms + offshore_windfarms + solarfarms

de_regions = [
    {'name':'DE_AMPRION','suffix':'_ampr', 'TSO':'Amprion'},
    {'name':'DE_50HZ','suffix':'_50hz', 'TSO':'50Hertz'},
    {'name':'DE_TENNET','suffix':'_tenn', 'TSO':'TenneT'},
    {'name':'DE_TRANSNET','suffix':'_tran', 'TSO':'TransnetBW'},
]