"""
WARNING!
Information given here may be inaccurate as open source data for all the installations and cities is not readily
available.
Moreover, it is often not clear which TSO is responsible for which wind or solar farm or even city.
In several cases I had to make and educated guess.

I had to collect the data from various sources to get a complete list.

Use it at your own risk.
Any verifications or corrections are highly appreciated!


Sources:
Locations and coordiantes: ChatGPT
Roughness length: https://www.wind101.net/wind-height/index.htm
Roughness factor: https://eurocodes-tools.com/en/roughness-factor-crz/

MAP energy companies: https://www.enet-navigator.de/aktuelles/uebertragungsnetzentgelte-preisentwicklungen-durchwachsen#&gid=2&pid=1

"""
from holidays.countries import France

''' -------------- GERMANY ---------------- '''

de_loc_cities = [
    # {
    #     "name": "Leipzig",
    #     "type": "city",
    #     "lat": None, # float; decimal latitude
    #     "lon": None, # gloat;  decimal longitude
    #     "population": None,  # int; Total population
    #     "population_density": None, # float; Persons per square kilometer (approx.)
    #     "area": None,  # float; Square kilometers
    #     "industrial_activity_fraction": None,  # float; Fraction of energy consumed by industry
    #     "renewable_energy_fraction": {
    #         "solar": None,  # float; Fraction of total energy from solar
    #         "wind": None,   # float; Fraction of total energy from wind
    #         "others": None  # float; Fraction of other renewable sources
    #     },
    #     "non_renewable_energy_fraction": None,  # float; Fraction of energy from non-renewables
    #     "total_energy_consumption": None,  # float; Annual energy consumption in GWh
    #     "peak_demand": None,  # float; Peak energy demand in MW
    #     "heating_degree_days": None,  # float; HDD (indicative of heating demand)
    #     "cooling_degree_days": None,   # float;  CDD (indicative of cooling demand)
    #     "installed_renewable_capacity": {
    #         "solar": None,  # float; Installed solar capacity in MW
    #         "wind": None    # float; Installed wind capacity in MW
    #     },
    #     "electric_vehicle_count": None,  # int; Number of EVs in the city
    #     "daylight_savings": True  # bool; Whether the city observes daylight savings
    # },

    # 50Hertz: Berlin, Rostok, Leipzig, Dresden (Hambug (partly))
    {
        "name": "Berlin",
        "label":'berlin',
        "type": "city",
        "suffix":"_city_berlin",
        "TSO": "50Hertz",
        "lat": 52.520007,
        "lon": 13.404954,
        "population": 3576873,  # Total population
        "population_density": 4000,  # Persons per square kilometer (approx.)
        "area": 891.8,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.12,  # Fraction of total energy from solar
            "wind": 0.18,   # Fraction of total energy from wind
            "others": 0.05  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.65,  # Fraction of energy from non-renewables
        "total_energy_consumption": 13500,  # Annual energy consumption in GWh
        "peak_demand": 1100,  # Peak energy demand in MW
        "avg_temperature": 9.8,  # Annual average temperature in °C
        "heating_degree_days": 2500,  # HDD (indicative of heating demand)
        "cooling_degree_days": 200,   # CDD (indicative of cooling demand)
        "avg_humidity": 75,  # Annual average relative humidity in %
        "avg_wind_speed": 13,  # Average wind speed in km/h
        "installed_renewable_capacity": {
            "solar": 500,  # Installed solar capacity in MW
            "wind": 750    # Installed wind capacity in MW
        },
        "electric_vehicle_count": 100000,  # Number of EVs in the city
        "timezone": "CET",
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Hamburg",
        "label":'hamburg',
        "type": "city",
        "suffix":"_city_hamburg",
        "TSO": "50Hertz", # NOTE that it is only partly in 50Hertz area
        "lat": 53.5506,  # decimal latitude
        "lon": 9.9933,   # decimal longitude
        "population": 1808846,  # Total population as of May 15, 2022
        "population_density": 2395,  # Persons per square kilometer
        "area": 755.22,  # Square kilometers
        "industrial_activity_fraction": 0.30,  # Approximate fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.02,  # Fraction of total energy from solar
            "wind": 0.10,   # Fraction of total energy from wind
            "others": 0.05  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.83,  # Fraction of energy from non-renewables
        "total_energy_consumption": 12_000,  # Annual energy consumption in GWh
        "peak_demand": 2_500,  # Peak energy demand in MW
        "heating_degree_days": 3000,  # HDD (indicative of heating demand)
        "cooling_degree_days": 100,   # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 100,  # Installed solar capacity in MW
            "wind": 500    # Installed wind capacity in MW
        },
        "electric_vehicle_count": 10_000,  # Number of EVs in the city
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Leipzig",
        "label":'leipzig',
        "type": "city",
        "suffix":"_city_leipzig",
        "TSO": "50Hertz",
        "lat": 51.3400,  # Latitude of Leipzig
        "lon": 12.3750,  # Longitude of Leipzig
        "population": 628718,  # Population as of 2023
        "population_density": 2100,  # Persons per square kilometer
        "area": 297.36,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Estimated based on regional data
        "renewable_energy_fraction": {
            "solar": 0.10,  # Estimated based on regional data
            "wind": 0.15,   # Estimated based on regional data
            "others": 0.05  # Estimated based on regional data
        },
        "non_renewable_energy_fraction": 0.70,  # Estimated based on regional data
        "total_energy_consumption": 12.5,  # Estimated in TWh/year
        "peak_demand": 2.5,  # Estimated in GW
        "heating_degree_days": 3000,  # Estimated based on regional data
        "cooling_degree_days": 100,  # Estimated based on regional data
        "installed_renewable_capacity": {
            "solar": 150,  # Estimated in MW
            "wind": 200    # Estimated in MW
        },
        "electric_vehicle_count": 5000,  # Estimated based on regional data
        "daylight_savings": True  # Leipzig observes daylight savings
    },
    {
        "name": "Dresden",
        "label":'dresden',
        "type": "city",
        "suffix":"_city_dresden",
        "TSO": "50Hertz",
        "lat": 51.0504,  # Corrected latitude for Dresden
        "lon": 13.7373,  # Corrected longitude for Dresden
        "population": 556780,  # Population as of 2023
        "population_density": 1872,  # Persons per square kilometer
        "area": 297.36,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Estimated based on regional data
        "renewable_energy_fraction": {
            "solar": 0.12,  # Estimated based on regional data
            "wind": 0.15,   # Estimated based on regional data
            "others": 0.08  # Estimated based on regional data
        },
        "non_renewable_energy_fraction": 0.65,  # Estimated based on regional data
        "total_energy_consumption": 7500,  # Estimated in GWh/year
        "peak_demand": 1200,  # Estimated in MW
        "heating_degree_days": 3400,  # Estimated based on regional climate data
        "cooling_degree_days": 150,  # Estimated based on regional climate data
        "installed_renewable_capacity": {
            "solar": 150,  # Estimated in MW
            "wind": 180    # Estimated in MW
        },
        "electric_vehicle_count": 5000,  # Estimated based on regional trends
        "daylight_savings": True  # Dresden observes daylight savings
    },
    {
        "name": "Magdeburg",
        "label":'magdeburg',
        "type": "city",
        "suffix":"_city_magdeburg",
        "TSO": "50Hertz",
        "lat": 52.1316,  # Corrected latitude for Magdeburg
        "lon": 11.6398,  # Corrected longitude for Magdeburg
        "population": 235723,  # Population as of 2023
        "population_density": 792,  # Persons per square kilometer
        "area": 297.36,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Estimated fraction of industrial activity
        "renewable_energy_fraction": {
            "solar": 0.10,  # Estimated fraction of energy from solar
            "wind": 0.25,   # Estimated fraction of energy from wind
            "others": 0.05  # Estimated fraction of energy from other renewable sources
        },
        "non_renewable_energy_fraction": 0.60,  # Estimated fraction of non-renewable energy
        "total_energy_consumption": 3500,  # Estimated total energy consumption in GWh
        "peak_demand": 500,  # Estimated peak demand in MW
        "heating_degree_days": 3000,  # Estimated heating degree days
        "cooling_degree_days": 100,  # Estimated cooling degree days
        "installed_renewable_capacity": {
            "solar": 50,  # Estimated installed solar capacity in MW
            "wind": 120   # Estimated installed wind capacity in MW
        },
        "electric_vehicle_count": 1500,  # Estimated number of electric vehicles
        "daylight_savings": True  # Magdeburg observes daylight savings
    },

    # Tennet: Munich, Bremen, Kiel, Kassel, Bremerhaven, Emden
    {
        "name": "Munich",
        "label":'munich',
        "type": "city",
        "suffix":"_city_munich",
        "TSO": "TenneT",
        "lat": 48.1351,  # Corrected latitude for Munich
        "lon": 11.5820,  # Corrected longitude for Munich
        "population": 1487708,  # Population as of 2023
        "population_density": 4800,  # Persons per square kilometer
        "area": 310.7,  # Square kilometers
        "industrial_activity_fraction": 0.20,  # Estimated fraction of industrial activity
        "renewable_energy_fraction": {
            "solar": 0.10,  # Estimated fraction of solar energy
            "wind": 0.05,   # Estimated fraction of wind energy
            "others": 0.15  # Estimated fraction of other renewable energies
        },
        "non_renewable_energy_fraction": 0.70,  # Estimated fraction of non-renewable energy
        "total_energy_consumption": 12200,  # GWh per year
        "peak_demand": 2_500,  # MW
        "heating_degree_days": 3_000,  # Degree days per year
        "cooling_degree_days": 100,  # Degree days per year
        "installed_renewable_capacity": {
            "solar": 150,  # MW
            "wind": 50     # MW
        },
        "electric_vehicle_count": 15000,  # Number of electric vehicles
        "daylight_savings": True  # Munich observes daylight savings
    },
    {
        "name": "Bremen",
        "label":'bremen',
        "type": "city",
        "suffix":"_city_bremen",
        "TSO": "TenneT",
        "lat": 53.0793,  # Corrected latitude for Bremen
        "lon": 8.8017,   # Corrected longitude for Bremen
        "population": 569352,  # Population as of 2023
        "population_density": 1914,  # Persons per square kilometer
        "area": 297.36,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Estimated based on regional economic data
        "renewable_energy_fraction": {
            "solar": 0.05,  # Estimated based on regional renewable energy data
            "wind": 0.15,   # Estimated based on regional renewable energy data
            "others": 0.05  # Estimated based on regional renewable energy data
        },
        "non_renewable_energy_fraction": 0.75,  # Estimated based on national energy consumption data
        "total_energy_consumption": 15_000_000,  # Estimated in megawatt-hours (MWh)
        "peak_demand": 2_500,  # Estimated in megawatts (MW)
        "heating_degree_days": 3_000,  # Estimated based on regional climate data
        "cooling_degree_days": 100,  # Estimated based on regional climate data
        "installed_renewable_capacity": {
            "solar": 50,  # Estimated in megawatts (MW)
            "wind": 150   # Estimated in megawatts (MW)
        },
        "electric_vehicle_count": 5_000,  # Estimated based on national trends and city size
        "daylight_savings": True  # Bremen observes daylight savings
    },
    {
        "name": "Kiel",
        "label":'kiel',
        "type": "city",
        "suffix":"_city_kiel",
        "TSO": "TenneT",
        "lat": 54.3233,  # Corrected latitude for Kiel
        "lon": 10.1228,  # Corrected longitude for Kiel
        "population": 246243,  # Population as of 2023
        "population_density": 1300,  # Persons per square kilometer
        "area": 118.65,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Estimated based on regional data
        "renewable_energy_fraction": {
            "solar": 0.10,  # Estimated based on regional data
            "wind": 0.30,   # Estimated based on regional data
            "others": 0.05  # Estimated based on regional data
        },
        "non_renewable_energy_fraction": 0.55,  # Estimated based on regional data
        "total_energy_consumption": 3000,  # Estimated in GWh/year
        "peak_demand": 500,  # Estimated in MW
        "heating_degree_days": 3000,  # Estimated based on regional climate data
        "cooling_degree_days": 100,  # Estimated based on regional climate data
        "installed_renewable_capacity": {
            "solar": 50,  # Estimated in MW
            "wind": 150   # Estimated in MW
        },
        "electric_vehicle_count": 2000,  # Estimated number of EVs
        "daylight_savings": True  # Kiel observes daylight savings
},
    {
        "name": "Kassel",
        "label":'kassel',
        "type": "city",
        "suffix":"_city_kassel",
        "TSO": "TenneT",
        "lat": 51.3127,  # decimal latitude
        "lon": 9.4797,   # decimal longitude
        "population": 201048,  # Total population as of December 2020
        "population_density": 1884,  # Persons per square kilometer
        "area": 106.8,  # Square kilometers
        "industrial_activity_fraction": 0.35,  # Estimated fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.05,  # Estimated fraction of total energy from solar
            "wind": 0.10,   # Estimated fraction of total energy from wind
            "others": 0.15  # Estimated fraction from other renewable sources
        },
        "non_renewable_energy_fraction": 0.70,  # Estimated fraction of energy from non-renewables
        "total_energy_consumption": 3500,  # Estimated annual energy consumption in GWh
        "peak_demand": 600,  # Estimated peak energy demand in MW
        "heating_degree_days": 3000,  # Estimated HDD
        "cooling_degree_days": 100,   # Estimated CDD
        "installed_renewable_capacity": {
            "solar": 50,  # Estimated installed solar capacity in MW
            "wind": 100   # Estimated installed wind capacity in MW
        },
        "electric_vehicle_count": 2000,  # Estimated number of EVs in the city
        "daylight_savings": True  # Kassel observes daylight savings
    },
    {
        "name": "Bremerhaven",
        "label":'bremerhaven',
        "type": "city",
        "suffix":"_city_bremerhaven",
        "TSO": "TenneT",
        "lat": 53.5396,  # decimal latitude
        "lon": 8.5809,  # decimal longitude
        "population": 113643,  # Total population
        "population_density": 1400,  # Persons per square kilometer (approx.)
        "area": 78.87,  # Square kilometers
        "industrial_activity_fraction": 0.35,  # Fraction of energy consumed by industry (estimated)
        "renewable_energy_fraction": {
            "solar": 0.05,  # Fraction of total energy from solar (estimated)
            "wind": 0.25,   # Fraction of total energy from wind (estimated)
            "others": 0.10  # Fraction of other renewable sources (estimated)
        },
        "non_renewable_energy_fraction": 0.60,  # Fraction of energy from non-renewables (estimated)
        "total_energy_consumption": 1200,  # Annual energy consumption in GWh (estimated)
        "peak_demand": 200,  # Peak energy demand in MW (estimated)
        "heating_degree_days": 3199,  # HDD (indicative of heating demand)
        "cooling_degree_days": 0,   # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 10,  # Installed solar capacity in MW (estimated)
            "wind": 50    # Installed wind capacity in MW (estimated)
        },
        "electric_vehicle_count": 500,  # Number of EVs in the city (estimated)
        "daylight_savings": True  # Whether the city observes daylight savings
    },

    # TransnetBW: Stuttgard, Karlsruhe, Mannheim, Freiburg im Breisgau, Heidelberg
    {
        "name": "Stuttgart",
        "label":'stuttgart',
        "type": "city",
        "suffix":"_city_stuttgart",
        "TSO": "TransnetBW",
        "lat": 48.7758,  # Decimal latitude
        "lon": 9.1829,   # Decimal longitude
        "population": 610458,  # Total population as of May 15, 2022
        "population_density": 2939,  # Persons per square kilometer
        "area": 207.36,  # Square kilometers
        "industrial_activity_fraction": 0.35,  # Estimated fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.05,  # Estimated fraction of total energy from solar
            "wind": 0.02,   # Estimated fraction of total energy from wind
            "others": 0.08  # Estimated fraction from other renewable sources
        },
        "non_renewable_energy_fraction": 0.85,  # Estimated fraction of energy from non-renewables
        "total_energy_consumption": 12000,  # Estimated annual energy consumption in GWh
        "peak_demand": 2000,  # Estimated peak energy demand in MW
        "heating_degree_days": 3400,  # HDD indicative of heating demand
        "cooling_degree_days": 450,   # CDD indicative of cooling demand
        "installed_renewable_capacity": {
            "solar": 150,  # Installed solar capacity in MW
            "wind": 50     # Installed wind capacity in MW
        },
        "electric_vehicle_count": 5000,  # Estimated number of EVs in the city
        "daylight_savings": True  # Stuttgart observes daylight savings
    },
    {
        "name": "Karlsruhe",
        "label":'karlsruhe',
        "type": "city",
        "suffix":"_city_karlsruhe",
        "TSO": "TransnetBW",
        "lat": 49.0069,  # decimal latitude
        "lon": 8.4037,   # decimal longitude
        "population": 305408,  # Total population as of May 15, 2022
        "population_density": 1800,  # Persons per square kilometer (approximation)
        "area": 173.5,  # Square kilometers
        "industrial_activity_fraction": 0.30,  # Fraction of energy consumed by industry (estimated)
        "renewable_energy_fraction": {
            "solar": 0.10,  # Fraction of total energy from solar (estimated)
            "wind": 0.05,   # Fraction of total energy from wind (estimated)
            "others": 0.15  # Fraction of other renewable sources (estimated)
        },
        "non_renewable_energy_fraction": 0.70,  # Fraction of energy from non-renewables (estimated)
        "total_energy_consumption": 2500,  # Annual energy consumption in GWh (estimated)
        "peak_demand": 450,  # Peak energy demand in MW (estimated)
        "heating_degree_days": 3000,  # HDD (indicative of heating demand, estimated)
        "cooling_degree_days": 100,    # CDD (indicative of cooling demand, estimated)
        "installed_renewable_capacity": {
            "solar": 50,  # Installed solar capacity in MW (estimated)
            "wind": 20    # Installed wind capacity in MW (estimated)
        },
        "electric_vehicle_count": 5000,  # Number of EVs in the city (estimated)
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Mannheim",
        "label":'mannheim',
        "type": "city",
        "suffix":"_city_mannheim",
        "TSO": "TransnetBW",
        "lat": 49.4875,  # float; decimal latitude
        "lon": 8.4660,  # float; decimal longitude
        "population": 313693,  # int; Total population
        "population_density": 2165.5,  # float; Persons per square kilometer (approx.)
        "area": 144.96,  # float; Square kilometers
        "industrial_activity_fraction": 0.35,  # float; Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.14,  # float; Fraction of total energy from solar
            "wind": 0.32,  # float; Fraction of total energy from wind
            "others": 0.14  # float; Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.4,  # float; Fraction of energy from non-renewables
        "total_energy_consumption": 5000,  # float; Annual energy consumption in GWh
        "peak_demand": 800,  # float; Peak energy demand in MW
        "heating_degree_days": 3199,  # float; HDD (indicative of heating demand)
        "cooling_degree_days": 100,  # float; CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 50,  # float; Installed solar capacity in MW
            "wind": 100  # float; Installed wind capacity in MW
        },
        "electric_vehicle_count": 5000,  # int; Number of EVs in the city
        "daylight_savings": True  # bool; Whether the city observes daylight savings
    },
    {
        "name": "Freiburg im Breisgau",
        "label":'freiburg_im_breisgau',
        "type": "city",
        "suffix":"_city_freiburg",
        "TSO": "TransnetBW",
        "lat": 47.999,  # decimal latitude
        "lon": 7.842,  # decimal longitude
        "population": 231195,  # Total population
        "population_density": 1509.8,  # Persons per square kilometer
        "area": 153.07,  # Square kilometers
        "industrial_activity_fraction": 0.25,  # Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.10,  # Fraction of total energy from solar
            "wind": 0.05,   # Fraction of total energy from wind
            "others": 0.15  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.70,  # Fraction of energy from non-renewables
        "total_energy_consumption": 2500,  # Annual energy consumption in GWh
        "peak_demand": 450,  # Peak energy demand in MW
        "heating_degree_days": 3000,  # HDD (indicative of heating demand)
        "cooling_degree_days": 450,   # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 50,  # Installed solar capacity in MW
            "wind": 20    # Installed wind capacity in MW
        },
        "electric_vehicle_count": 1500,  # Number of EVs in the city
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Heidelberg",
        "label":'heidelberg',
        "type": "city",
        "suffix":"_city_heidelberg",
        "TSO": "TransnetBW",
        "lat": 49.3988,  # Latitude for Heidelberg, Germany
        "lon": 8.6724,   # Longitude for Heidelberg, Germany
        "population": 160000,  # Population as of 2019
        "population_density": 1470,  # Approximate population density (persons per square kilometer)
        "area": 108.8,  # Area in square kilometers
        "industrial_activity_fraction": 0.12,  # Estimated fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.05,  # Estimated fraction of total energy from solar
            "wind": 0.02,   # Estimated fraction of total energy from wind
            "others": 0.08  # Estimated fraction from other renewable sources
        },
        "non_renewable_energy_fraction": 0.85,  # Estimated fraction of energy from non-renewables
        "total_energy_consumption": 1200,  # Estimated annual energy consumption in GWh
        "peak_demand": 250,  # Estimated peak energy demand in MW
        "heating_degree_days": 3000,  # Estimated HDD for Heidelberg
        "cooling_degree_days": 100,   # Estimated CDD for Heidelberg
        "installed_renewable_capacity": {
            "solar": 15,  # Estimated installed solar capacity in MW
            "wind": 5     # Estimated installed wind capacity in MW
        },
        "electric_vehicle_count": 2000,  # Estimated number of EVs in the city
        "daylight_savings": True  # Heidelberg observes daylight savings
    },

    # Amprion: Frankfurt am Main, Cologne, Düsseldorf, Essen, Dortmund
    {
        "name": "Frankfurt am Main",
        "label":'frankfurt_am_main',
        "type": "city",
        "suffix":"_city_frankfurt",
        "TSO": "Amprion",
        "lat": 50.1109,  # Decimal latitude
        "lon": 8.6821,   # Decimal longitude
        "population": 753056,  # Total population as of 2019
        "population_density": 3033.3,  # Persons per square kilometer
        "area": 248.31,  # Square kilometers
        "industrial_activity_fraction": 0.30,  # Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.05,  # Fraction of total energy from solar
            "wind": 0.10,   # Fraction of total energy from wind
            "others": 0.20  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.65,  # Fraction of energy from non-renewables
        "total_energy_consumption": 22600,  # Annual energy consumption in GWh as of 2010
        "peak_demand": 1500,  # Peak energy demand in MW
        "heating_degree_days": 3000,  # HDD (indicative of heating demand)
        "cooling_degree_days": 450,   # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 50,  # Installed solar capacity in MW
            "wind": 85    # Installed wind capacity in MW
        },
        "electric_vehicle_count": 5000,  # Number of EVs in the city
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Cologne",
        "label":'cologne',
        "type": "city",
        "suffix":"_city_cologne",
        "TSO": "Amprion",
        "lat": 50.9423,  # Decimal latitude
        "lon": 6.9570,   # Decimal longitude
        "population": 1149010,  # Total population
        "population_density": 2835,  # Persons per square kilometer
        "area": 405.15,  # Square kilometers
        "industrial_activity_fraction": 0.29,  # Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.05,  # Fraction of total energy from solar
            "wind": 0.02,   # Fraction of total energy from wind
            "others": 0.08  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.85,  # Fraction of energy from non-renewables
        "total_energy_consumption": 18500,  # Annual energy consumption in GWh
        "peak_demand": 2500,  # Peak energy demand in MW
        "heating_degree_days": 2908,  # HDD (indicative of heating demand)
        "cooling_degree_days": 140,    # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 150,  # Installed solar capacity in MW
            "wind": 50     # Installed wind capacity in MW
        },
        "electric_vehicle_count": 5000,  # Number of EVs in the city
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Düsseldorf",
        "label":'duesseldorf',
        "type": "city",
        "suffix":"_city_duesseldorf",
        "TSO": "Amprion",
        "lat": 51.2277,  # Decimal latitude
        "lon": 6.7735,  # Decimal longitude
        "population": 646000,  # Total population
        "population_density": 2860,  # Persons per square kilometer
        "area": 217,  # Square kilometers
        "industrial_activity_fraction": 0.30,  # Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.07,  # Fraction of total energy from solar
            "wind": 0.23,   # Fraction of total energy from wind
            "others": 0.25  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.45,  # Fraction of energy from non-renewables
        "total_energy_consumption": 12000,  # Annual energy consumption in GWh
        "peak_demand": 1500,  # Peak energy demand in MW
        "heating_degree_days": 2800,  # HDD (indicative of heating demand)
        "cooling_degree_days": 350,   # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 150,  # Installed solar capacity in MW
            "wind": 200    # Installed wind capacity in MW
        },
        "electric_vehicle_count": 5000,  # Number of EVs in the city
        "daylight_savings": True  # Whether the city observes daylight savings
    },
    {
        "name": "Essen",
        "label":'essen',
        "type": "city",
        "suffix":"_city_essen",
        "TSO": "Amprion",
        "lat": 51.4508,  # decimal latitude
        "lon": 7.0131,  # decimal longitude
        "population": 571039,  # Total population as of May 15, 2022
        "population_density": 2714,  # Persons per square kilometer
        "area": 210.34,  # Square kilometers
        "industrial_activity_fraction": 0.30,  # Estimated fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.05,  # Estimated fraction of total energy from solar
            "wind": 0.10,   # Estimated fraction of total energy from wind
            "others": 0.15  # Estimated fraction from other renewable sources
        },
        "non_renewable_energy_fraction": 0.70,  # Estimated fraction of energy from non-renewables
        "total_energy_consumption": 5000,  # Estimated annual energy consumption in GWh
        "peak_demand": 1000,  # Estimated peak energy demand in MW
        "heating_degree_days": 3000,  # HDD indicative of heating demand
        "cooling_degree_days": 100,   # CDD indicative of cooling demand
        "installed_renewable_capacity": {
            "solar": 50,  # Installed solar capacity in MW
            "wind": 100   # Installed wind capacity in MW
        },
        "electric_vehicle_count": 5000,  # Estimated number of EVs in the city
        "daylight_savings": True  # Essen observes daylight savings
    },
    {
        "name": "Dortmund",
        "label":'dortmund',
        "type": "city",
        "suffix":"_city_dortmund",
        "TSO": "Amprion",
        "lat": 51.5136,  # decimal latitude
        "lon": 7.4653,   # decimal longitude
        "population": 598246,  # Total population as of May 15, 2022
        "population_density": 2131.5,  # Persons per square kilometer
        "area": 280.4,  # Square kilometers
        "industrial_activity_fraction": 0.29,  # Fraction of energy consumed by industry
        "renewable_energy_fraction": {
            "solar": 0.10,  # Fraction of total energy from solar
            "wind": 0.15,   # Fraction of total energy from wind
            "others": 0.05  # Fraction of other renewable sources
        },
        "non_renewable_energy_fraction": 0.70,  # Fraction of energy from non-renewables
        "total_energy_consumption": 5000,  # Annual energy consumption in GWh
        "peak_demand": 800,  # Peak energy demand in MW
        "heating_degree_days": 3000,  # HDD (indicative of heating demand)
        "cooling_degree_days": 100,   # CDD (indicative of cooling demand)
        "installed_renewable_capacity": {
            "solar": 4.2,  # Installed solar capacity in MW
            "wind": 10.0   # Installed wind capacity in MW
        },
        "electric_vehicle_count": 2000,  # Number of EVs in the city
        "daylight_savings": True  # Whether the city observes daylight savings
    }
]

de_loc_onshore_windfarms = [
    # 50Hertz
    {
        "name": "Hüselitz Wind Farm",
        "label":"hueselitz_wind_farm",
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
        "label":"werder_kessin_wind_farm",
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
        "label":"uckermark_enertrag_wind_farm",
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
        "label":"feldheim_wind_farm",
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
        "label":"reussenkoege_wind_farm",
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
        "label":"jade_wind_farm",
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
        "label":"veer_doerper_wind_farm",
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
        "label":"hohenlochen_wind_farm",
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
        "label":"harthaeuser_wald_wind_farm",
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
        "label":"straubenhardt_wind_farm",
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
        "label":"hollich_wind_farm",
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
        "label":"coesfeld_letter_bruch_wind_farm",
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
        "label":"a31_gescher_reken_wind_farm",
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

de_loc_offshore_windfarms = [
    # TenneT
    {
        "name": "EnBW Hohe See",
        "label":"enbw_hohe_see_wind_farm",
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
        "label":"borkum_riffgrund_wind_farm",
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
        "label":"veja_mate_wind_farm",
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
        "label":"bard_offshore_1_wind_farm",
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
        "label":"global_tech_i_wind_farm",
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
        "label":"wikinger_offshore_wind_farm",
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
        "label":"arkona_wind_farm",
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

de_loc_solarfarms = [
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
        "label":"witznitz_solar_farm",
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
        "label":"weesow_willmersdorf_solar_farm",
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
        "label":"tramm_Goethen_solar_farm",
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
        "label":"meuro_solar_farm",
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
        "label":"gottesgabe_solar_farm",
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
        "label":"ernsthof_solar_farm",
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
        "label":"erbach_solar_farm",
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
        "label":"aalen_solar_farm",
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
        "label":"lauingen_solar_farm",
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
        "label":"schlechtenberg_solar_farm",
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
        "TSO": "TenneT", # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_schlechtenberg",
        "type": "solar farm"
    },
    {
        "name": "Mengkofen Solar Park",
        "label":"mengkofen_solar_farm",
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
        "label":"eggebek_solar_farm",
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
        "label":"bottrop_solar_farm",
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
        "label":"kirchhellen_solar_farm",
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
        "label":"nettersheim_solar_farm",
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
        "label":"frankfurt_solar_farm",
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
        "TSO": "Amprion",  # str; TSO that connects this solar farm to the grid
        "suffix": "_sol_frankfurt",
        "type": "solar farm",
    }

]

de_all_locations = de_loc_cities + de_loc_solarfarms + de_loc_offshore_windfarms + de_loc_onshore_windfarms


''' ----------- FRANCE ------------- '''

fr_loc_onshore_windfarms = [
    # RTE
    {
        "name": "Ensemble Eolien Catalan Wind Farm",
        "label": "eolien_catalan_wind_farm",
        "capacity": 96.0,  # MW
        "n_turbines": 35,  # total number of turbines
        "lat": 42.7274,  # decimal degrees
        "lon": 2.7786,  # decimal degrees
        "elevation": 100,  # meters; approximate elevation above sea level
        "z0": 0.1,  # meters; roughness length
        "terrain_category": "II",  # as defined in the Eurocode standards
        "location": "Pyrénées-Orientales, Occitanie, France",
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_won_eolien",
        "type": "onshore wind farm",
        "link": "https://edf-renouvelables.com/en/projet/ensemble-eolien-catalan/"
    },
    {
        "name": "Salles-Curan Wind Farm",
        "label": "salles_curan_wind_farm",
        "capacity": 87.0,  # float MW
        "n_turbines": 29,  # int total number of turbines
        "lat": 44.1633,  # float, decimal degrees
        "lon": 2.8282,  # float, decimal degrees
        "elevation": 800.0,  # float, meters; approximate elevation above sea level
        "z0": 0.03,  # float, meters; roughness length
        "terrain_category": "II",  # str, as defined in the Eurocode standards
        "location": "Salles-Curan, Aveyron, Occitanie",
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_won_salles",
        "type": "onshore wind farm",
        "link": "https://www.gem.wiki/Salles-Curan_wind_farm"
    },
    {
        "name": "Les Hauts Pays Wind Farm",
        "label": "les_hauts_pays_wind_farm",
        "capacity": 80.0,  # float MW
        "n_turbines": 39,  # int total number of turbines
        "lat": 48.3835,  # float, decimal degrees
        "lon": 5.3403,  # float, decimal degrees
        "elevation": 140.0,  # float, meters; approximate elevation above sea level
        "z0": 0.3,  # float, meters; roughness length
        "terrain_category": "III",  # str, as defined in the Eurocode standards
        "location": "Epizon, Haute-Marne, Grand Est, France",  # str, (county or city)
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_won_leshauts",
        "type": "onshore wind farm",
        "link": "https://www.gem.wiki/Les_Hauts_Pays_wind_farm"  # main source of information
    },
    {
        "name": "Champagne Picardie Project Wind Farm",
        "label": "champagne_picardie_wind_farm",
        "capacity": 72.6,  # float MW
        "n_turbines": 22,  # int total number of turbines
        "lat": 49.6406,  # float, decimal degrees
        "lon": 3.8644,  # float, decimal degrees
        "elevation": 100,  # float, meters; approximate elevation above sea level
        "z0": 0.05,  # float, meters; roughness length
        "terrain_category": "II",  # str, as defined in the Eurocode standards
        "location": "Bucy-lès-Pierrepont, Aisne, Hauts-de-France",  # str, (county or city)
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_won_champagne",
        "type": "onshore wind farm",
        "link": "https://edf-renouvelables.com/en/projet/champagne-picarde/"
    },
    {
        "name": "Mont Payard Wind Farm",
        "label": "mont_payard_wind_farm",
        "capacity": 75.0,  # float MW
        "n_turbines": 30,  # int total number of turbines
        "lat": 48.894167,  # float, decimal degrees
        "lon": 4.194611,  # float, decimal degrees
        "elevation": 150.0,  # float, meters; approximate elevation above sea level
        "z0": 0.03,  # float, meters; roughness length
        "terrain_category": "II",  # str, as defined in the Eurocode standards
        "location": "Germinon, Vélye, Grand Est, France",  # str, (county or city)
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_won_mont",
        "type": "onshore wind farm",
        "link": "https://www.power-technology.com/data-insights/power-plant-profile-mont-payard-france/"
    }
]

fr_loc_offshore_windfarms = [
    {
        "name": "Fécamp Offshore Wind Farm",
        "label": "feecamp_wind_farm",
        "capacity": 497.0,  # float MW
        "n_turbines": 71,  # int total number of turbines
        "lat": 49.9812,  # float, decimal degrees
        "lon": 0.4538,  # float, decimal degrees
        "location": "Fécamp, Seine-Maritime",  # str, (county or city)
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_woff_feecamp",
        "type": "offshore wind farm",
        "link": "https://www.enbridge.com/projects-and-infrastructure/projects/fecamp-offshore-wind-project"
    },
    {
        "name": "Saint-Brieuc Offshore Wind Farm",
        "label": "saint_brieuc_wind_farm",
        "capacity": 496.0,  # float MW
        "n_turbines": 62,  # int total number of turbines
        "lat": 48.852,  # float, decimal degrees
        "lon": -2.539,  # float, decimal degrees
        "location": "Brittany",  # str, (county or city)
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_woff_saintb",
        "type": "offshore wind farm",
        "link": "https://www.iberdrola.com/press-room/news/detail/iberdrola-comissions-saint-brieuc-the-second-offshore-wind-farm-in-france-and-the-first-in-britanny"
    },
    {
        "name": "Saint-Nazaire Offshore Wind Farm",
        "label": "saint_nazaire_wind_farm",
        "capacity": 480.0,  # float MW
        "n_turbines": 80,  # int total number of turbines
        "lat": 47.160459,  # float, decimal degrees
        "lon": -2.626559,  # float, decimal degrees
        "location": "Loire-Atlantique",  # str, (county or city)
        "TSO": "RTE",  # France's transmission system operator
        "suffix": "_woff_saintn",
        "type": "offshore wind farm",
        "link": "https://parc-eolien-en-mer-de-saint-nazaire.fr/"  # main source of information
    }
]

fr_loc_solarfarms = [
    {
        "name": "Cestas Solar Park",
        "label": "cestas_solar_farm",
        "capacity": 300.0,  # float, in MW
        "n_panels": 983500,  # int, number of panels
        "type_panels": "Polycrystalline silicon modules",  # str, panel type
        "elevation": 60.0,  # float, meters; approximate elevation above sea level
        "z0": 0.3,  # float, meter (roughness length)
        "terrain_category": "II",  # str, as defined in the Eurocode standards ("I", "II", "III")
        "location": "Cestas, Gironde",
        "size": 260,  # hectares (size of the farm)
        "lat": 44.7262,  # decimal latitude
        "lon": -0.8077,  # decimal longitude
        "TSO": "RTE",  # France only has one TSO
        "suffix": "_sol_cestas",
        "type": "solar farm",
    },
    {
        "name": "Athies-Samoussy Solar Park",
        "label": "athies_samoussy_solar_farm",
        "capacity": 87.5,  # float, in MW
        "n_panels": 220000,  # int, number of panels
        "type_panels": "Photovoltaic (PV) modules",  # str, panel type
        "elevation": 80.0,  # float, meters; approximate elevation above sea level
        "z0": 0.03,  # float, meter (roughness length)
        "terrain_category": "II",  # str, as defined in the Eurocode standards ("I", "II", "III")
        "location": "Athies-sous-Laon and Samoussy, Hauts-de-France",
        "size": 100,  # hectares (size of the farm)
        "lat": 49.5879,  # decimal latitude
        "lon": 3.7003,  # decimal longitude
        "TSO": "RTE",  # France only has one TSO
        "suffix": "_sol_athies",
        "type": "solar farm",
    },
    {
        "name": "Toul-Rosières Solar Park",
        "label": "toul_rosieres_solar_farm",
        "capacity": 115.0,  # float, in MW
        "n_panels": 1400000,  # int, number of panels
        "type_panels": "Thin-film CdTe photovoltaic panels",  # str, panel type
        "elevation": 270.0,  # float, meters; approximate elevation above sea level
        "z0": 0.03,  # float, meter (roughness length)
        "terrain_category": "II",  # str, as defined in the Eurocode standards ("I", "II", "III")
        "location": "Rosières-en-Haye, Meurthe-et-Moselle",
        "size": 367,  # hectares (size of the farm)
        "lat": 48.785378,  # decimal latitude
        "lon": 5.979309,  # decimal longitude
        "TSO": "RTE",  # France only has one TSO
        "suffix": "_sol_toul",
        "type": "solar farm",
    },
    {
        "name": "Gabardan Solar Park",
        "label": "gabardan_solar_farm",
        "capacity": 67.5,  # float, in MW
        "n_panels": 872300,  # int, number of panels
        "type_panels": "Thin-film photovoltaic panels made by First Solar",  # str, panel type
        "elevation": 800,  # float, meters; approximate elevation above sea level
        "z0": None,  # float, meter (roughness length)
        "terrain_category": None,  # str, as defined in the Eurocode standards ("I", "II", "III")
        "location": "Losse, Landes, Nouvelle-Aquitaine, France",  # str (county or city)
        "size": 317,  # hectares (size of the farm)
        "lat": 44.061,  # decimal latitude
        "lon": -0.014,  # decimal longitude
        "TSO": "RTE",  # France only has one TSO
        "suffix": "_sol_gabardan",
        "type": "solar farm",
    }
]

fr_loc_cities = [
    {
        "name": "Paris",
        "label": "paris",
        "type": "city",
        "suffix": "_city_paris",
        "TSO": "RTE",
        "lat": 48.8566,
        "lon": 2.3522,
        "population": 2148000,
        "population_density": 25200,
        "area": 105.4,
        "industrial_activity_fraction": 0.1,
        "renewable_energy_fraction": {
            "solar": 0.05,
            "wind": 0.02,
            "others": 0.08
        },
        "non_renewable_energy_fraction": 0.85,
        "total_energy_consumption": 208000,
        "peak_demand": 5000,
        "avg_temperature": 12.3,
        "heating_degree_days": 2400,
        "cooling_degree_days": 450,
        "avg_humidity": 75,
        "avg_wind_speed": 15,
        "installed_renewable_capacity": {
            "solar": 150,
            "wind": 50
        },
        "electric_vehicle_count": 50000,
        "timezone": "CET",
        "daylight_savings": True,
        "nearest_nuclear_power_plant": {
            "name": "Nogent Nuclear Power Plant",
            "distance_from_city": 120,
            "power": 2620,
            "note": "Located in Nogent-sur-Seine, this plant houses two reactors with a combined capacity of around 2,620 MW"
        }
    },
    {
        "name": "Marseille",
        "label": "marseille",
        "type": "city",
        "suffix": "_city_marseille",
        "TSO": "RTE",
        "lat": 43.2965,
        "lon": 5.3698,
        "population": 877215,
        "population_density": 3646,
        "area": 240.62,
        "industrial_activity_fraction": 0.30,
        "renewable_energy_fraction": {
            "solar": 0.04,
            "wind": 0.08,
            "others": 0.10
        },
        "non_renewable_energy_fraction": 0.78,
        "total_energy_consumption": 5000,
        "peak_demand": 1000,
        "avg_temperature": 15.5,
        "heating_degree_days": 1500,
        "cooling_degree_days": 500,
        "avg_humidity": 70,
        "avg_wind_speed": 20,
        "installed_renewable_capacity": {
            "solar": 200,
            "wind": 150
        },
        "electric_vehicle_count": 5000,
        "timezone": "CET",
        "daylight_savings": True,
        "nearest_nuclear_power_plant": {
            "name": "Tricastin Nuclear Power Plant",
            "distance_from_city": 140,
            "power": 3660,
            "note": "Situated near the Rhône River, Tricastin comprises four reactors with a total capacity of about 3,660 MW"
        }
    },
    {
        "name": "Lyon",
        "label": "lyon",
        "type": "city",
        "suffix": "_city_lyon",
        "TSO": "RTE",
        "lat": 45.7578,
        "lon": 4.8322,
        "population": 522228,
        "population_density": 10908,
        "area": 47.87,
        "industrial_activity_fraction": 29.1,
        "renewable_energy_fraction": {
            "solar": 0.12,
            "wind": 0.05,
            "others": 0.10
        },
        "non_renewable_energy_fraction": 0.73,
        "total_energy_consumption": 8000,  # in GWh
        "peak_demand": 1500,  # in MW
        "avg_temperature": 12.0,
        "heating_degree_days": 2800,
        "cooling_degree_days": 450,
        "avg_humidity": 76,
        "avg_wind_speed": 12.2,
        "installed_renewable_capacity": {
            "solar": 20,
            "wind": 10
        },
        "electric_vehicle_count": 25000,
        "timezone": "CET",
        "daylight_savings": True,
        "nearest_nuclear_power_plant": {
            "name": "Bugey Nuclear Power Plant",
            "distance_from_city": 35,
            "power": 3580,
            "note": "Located in Saint-Vulbas, Bugey has four operational reactors with a combined capacity of around 3,580 MW"
        }
    },
    {
        "name": "Toulouse",
        "label": "Toulouse",
        "type": "city",
        "suffix": "_city_toulouse",
        "TSO": "RTE",
        "lat": 43.6045,
        "lon": 1.444,
        "population": 511684,
        "population_density": 3893,
        "area": 118.3,
        "industrial_activity_fraction": 0.086,
        "renewable_energy_fraction": {
            "solar": 0.03,
            "wind": 0.00,
            "others": 0.07
        },
        "non_renewable_energy_fraction": 0.90,
        "total_energy_consumption": 19350,
        "peak_demand": None, # unknown
        "avg_temperature": 15.2,
        "heating_degree_days": 2000,
        "cooling_degree_days": 500,
        "avg_humidity": 70,
        "avg_wind_speed": 14.3,
        "installed_renewable_capacity": {
            "solar": 15,
            "wind": 0
        },
        "electric_vehicle_count": 5000,
        "timezone": "CET",
        "daylight_savings": True,
        "nearest_nuclear_power_plant": {
            "name": "Golfech Nuclear Power Plant",
            "distance_from_city": 90,
            "power": 2620,
            "note": "This facility consists of two reactors with a total capacity of about 2,620 MW."
        }
    },
    {
        "name": "Lille",
        "label": "Lille",
        "type": "city",
        "suffix": "_city_lille",
        "TSO": "RTE",
        "lat": 50.6372,
        "lon": 3.0633,
        "population": 236234,
        "population_density": 6788.9,
        "area": 34.8,
        "industrial_activity_fraction": 0.25,
        "renewable_energy_fraction": {
            "solar": 0.05,
            "wind": 0.10,
            "others": 0.45
        },
        "non_renewable_energy_fraction": 0.4,
        "total_energy_consumption": 732,
        "peak_demand": 200,
        "avg_temperature": 11.0,
        "heating_degree_days": 2800,
        "cooling_degree_days": 50,
        "avg_humidity": 77,
        "avg_wind_speed": 17.7,
        "installed_renewable_capacity": {
            "solar": 50,
            "wind": 100
        },
        "electric_vehicle_count": 5000,
        "timezone": "CET",
        "daylight_savings": True,
        "nearest_nuclear_power_plant": {
            "name": "Gravelines Nuclear Power Plant",
            "distance_from_city": 70,
            "power": 5460,
            "note": "Gravelines is one of France's largest nuclear facilities, housing six reactors with a combined capacity of around 5,460 MW."
        }
    },
    # {
    #     "name": "Lille",
    #     "label":'Lille',
    #     "type": "city",
    #     "suffix":"_city_lille",
    #     "TSO": "RTE", # France only has one TSO
    #     "lat": None, # float; decimal latitude;
    #     "lon": None, # float; decimal latitude;
    #     "population": None,  # int; Total population
    #     "population_density": None,  # float; Persons per square kilometer (approx.)
    #     "area": None,  # float; Square kilometers
    #     "industrial_activity_fraction":None,  # float; Fraction of energy consumed by industry
    #     "renewable_energy_fraction": {
    #         "solar": None,  # float; Fraction of total energy from solar
    #         "wind": None,   # float; Fraction of total energy from wind
    #         "others": None  # float; Fraction of other renewable sources
    #     },
    #     "non_renewable_energy_fraction": None,  # float; Fraction of energy from non-renewables
    #     "total_energy_consumption": None,  # float; Annual energy consumption in GWh
    #     "peak_demand": None,  # float; Peak energy demand in MW
    #     "avg_temperature": None,  # float; Annual average temperature in °C
    #     "heating_degree_days": None,  # int; HDD (indicative of heating demand)
    #     "cooling_degree_days": None,  # int; CDD (indicative of cooling demand)
    #     "avg_humidity": None,  # float; Annual average relative humidity in %
    #     "avg_wind_speed": None,  # float; Average wind speed in km/h
    #     "installed_renewable_capacity": {
    #         "solar": None,  # float; Installed solar capacity in MW
    #         "wind": None    # float; Installed wind capacity in MW
    #     },
    #     "electric_vehicle_count": None,  # int; Number of EVs in the city
    #     "timezone": "CET",
    #     "daylight_savings": True,  # Whether the city observes daylight savings
    #     "nearest_nuclear_power_plant":{
    #         "name":"Gravelines Nuclear Power Plant",
    #         "distance_from_city":70,#km
    #         "power": 5460, # MW
    #         "note":"Gravelines is one of France's largest nuclear facilities, housing six reactors with a combined capacity of around 5,460 MW."
    #     }
    # },
]


''' -------------- SUMMARY -------------- '''

# de_regions = [
#     {'name':'DE_AMPRION', 'suffix':'_ampr', 'TSO':'Amprion'},
#     {'name':'DE_50HZ', 'suffix':'_50hz', 'TSO':'50Hertz'},
#     {'name':'DE_TENNET', 'suffix':'_tenn', 'TSO':'TenneT'},
#     {'name':'DE_TRANSNET', 'suffix':'_tran', 'TSO':'TransnetBW'},
# ]
country_code_name_mapping = {
    # --- Germany
    'AT':'austria', 'BE':'belgium', 'CH':'switzerland', 'CZ':'czechia',
    'DK_1':'denmark1', 'DK_2':'denmark2', 'FR':'france', 'NO_2':'norway2',
    'NL':'netherlands', 'PL':'poland', 'SE_4':'sweden4',
    # --- France
    'DE_AT_LU':'germany_old', 'DE_LU':'germany', 'ES':'estonia', 'GB':'greatbritain',
    'IT_NORD':'italynorth', 'IT_NORD_FR':'italynorthold'
}
countries_metadata = [
    # ------------ GERMANY --------------
    {
        'name':'Germany',
        'code':'DE',
        'bidding_zone':'DE_LU',
        'regions':[
            {'name':'DE_AMPRION', 'suffix':'_ampr', 'TSO':'Amprion',
                "available_targets":["wind_onshore","solar","load","energy_mix"]},
            {'name':'DE_50HZ', 'suffix':'_50hz', 'TSO':'50Hertz',
                "available_targets":["wind_offshore","wind_onshore","solar","load","energy_mix"]},
            {'name':'DE_TENNET', 'suffix':'_tenn', 'TSO':'TenneT',
                "available_targets":["wind_offshore","wind_onshore","solar","load","energy_mix"]},
            {'name':'DE_TRANSNET', 'suffix':'_tran', 'TSO':'TransnetBW',
                "available_targets":["wind_offshore","wind_onshore","solar","load","energy_mix"]},
        ],
        # "entsoe_neighbors":["AT","BE","CH","CZ","DK_1","DK_2","FR","NO_2","NL","PL","SE_4"],
        "locations":{
            "cities":de_loc_cities,
            "solar":de_loc_solarfarms,
            "offshore":de_loc_offshore_windfarms,
            "onshore":de_loc_onshore_windfarms,
        },
    },
    # ------------- FRANCE --------------
    {
        'name':'France',
        'code':'FR',
        'bidding_zone':'FR',
        'regions':[
            {'name':'RTE', 'suffix':'_rte', 'TSO':'RTE',
             "available_targets":["wind_offshore","wind_onshore","solar","load","energy_mix"]},
        ],
        'locations': {
            "cities":fr_loc_cities,
            "solar":fr_loc_solarfarms,
            "offshore":fr_loc_offshore_windfarms,
            "onshore":fr_loc_onshore_windfarms,
        }
    }
]

all_locations = []
for country in countries_metadata:
    for key, locs in country['locations'].items():
        all_locations.extend(locs)