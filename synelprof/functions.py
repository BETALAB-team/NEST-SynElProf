import time
import os
import glob

from scipy.stats import beta
import pandas as pd
import numpy as np
from synelprof.electric_load_italian_distribution import get_italian_random_el_consumption

def average_profile_creation_from_ARERA(
                        provincia,
                       giorno_iniziale,
                        time_step,
                       mercato = "Tutti",
                       fp = "FP3",
                       residenza = 'Tutti'
                       ):
    """
    This function returns a matrix which is 24*12 lines (5 timestep minutes per day) and 365 columns
    It represents the province annual average ARERA load profile

    Parameters
    ----------
    provincia: str
        Province for whìch the matrix is calculated
    giorno_iniziale: int
        initial day number (e.g. 3 for thursday)
    mercato: str, default 'Tutti'.
        market
    fp: str, default 'FP3'
        Power range of the contract
    residenza: str, default 'Tutti'
        select houses if resident or not

    Returns
    -------
    numpy.array
        A numpy array 24*12 rows by 365 columns
    """

    data_ARERA = pd.read_csv(os.path.join("synelprof", "Dati_orari_ARERA_2021.csv"), index_col = [0, 1, 2, 3, 4], header = [0, 1], sep =";")
    selected_province = pd.DataFrame(0, index = pd.date_range("00:00", periods = 24, freq = "1h"),columns = pd.MultiIndex.from_product([np.arange(12),["Weekday", "Sunday", "Saturday"]]))

    months ={
             "gen-21": [0,31],
             "feb-21": [1,28],
             "mar-21": [2,31],
             "apr-21": [3,30],
             "mag-21": [4,31],
             "giu-21": [5,30],
             "lug-21": [6,31],
             "ago-21": [7,31],
             "set-21": [8,30],
             "ott-21": [9,31],
             "nov-21": [10,30],
             "dic-21": [11,31],
             }

    distribution = np.zeros([24, 365])
    start_week_day = giorno_iniziale
    initial_day_number = 0
    final_day_number = 0
    for m_k, [m_nb, day_number] in months.items():
        final_day_number += day_number
        selected_province[m_nb,"Weekday"] = data_ARERA["Giorno feriale"].loc[provincia, m_k, mercato, fp, residenza].iloc[0:24].values
        selected_province[m_nb,"Saturday"] = data_ARERA["Sabato"].loc[provincia, m_k, mercato, fp, residenza].iloc[0:24].values
        selected_province[m_nb,"Sunday"] = data_ARERA["Domenica"].loc[provincia, m_k, mercato, fp, residenza].iloc[0:24].values

        week = np.hstack([
            np.repeat(selected_province[m_nb, "Weekday"].values.reshape(-1, 1), 5, axis=1),
            np.repeat(selected_province[m_nb, "Saturday"].values.reshape(-1, 1), 1, axis=1),
            np.repeat(selected_province[m_nb, "Sunday"].values.reshape(-1, 1), 1, axis=1)
        ])

        month = np.hstack([week, week, week, week, week, week])

        distribution[:,initial_day_number:final_day_number] = month[:,start_week_day:(start_week_day + day_number)]

        start_week_day = (start_week_day + day_number)%7
        initial_day_number = initial_day_number + day_number

    distribution = np.vstack([distribution, distribution[-1,:]])
    distribution = pd.DataFrame(distribution, index=pd.date_range("00:00", periods = 25, freq = "1h")).resample(time_step).interpolate("linear").values
    distribution = distribution[:-1,:]

    return distribution

def get_standard_app_profiles_f(time_step):
    """
    Returns a dictionary of dataframes with
    Parameters
    ----------
    time_step: str
        string to define the resampling time step (e.g. 5min)

    Returns
    -------
    dict
        Dictionary of pandas DataFrame

    """

    # Read loads standard profiles
    loads_standard_profiles = {}
    parquet_files = glob.glob(os.path.join("", "profiles_df_flx", '*.{}'.format('parquet')))
    for f in parquet_files:
        df = pd.read_parquet(f).loc(axis = 1)[:,"P"].fillna(0)
        df.columns = df.columns.droplevel(level = 1)
        loads_standard_profiles[f.split(os.sep)[-1][:-8]]  = df.set_index(pd.date_range(start="00:00", freq="1s", periods = len(df.index))).resample(time_step).mean()
        # fig, ax = plt.subplots(figsize = (10,10))
        # loads_standard_profiles[f.split(os.sep)[-1][:-8]].plot(ax = ax)
        # fig.savefig(os.path.join("synelprof","plots", f'{f.split(os.sep)[-1][:-8]}.png'))
        # plt.close()
    return loads_standard_profiles

def get_standard_app_profiles(time_step):
    """
    Returns a dictionary of dataframes with
    Parameters
    ----------
    time_step: str
        string to define the resampling time step (e.g. 5min)

    Returns
    -------
    dict
        Dictionary of pandas DataFrame

    """

    # Read loads standard profiles
    loads_standard_profiles = {}
    parquet_files = glob.glob(os.path.join("synelprof", "profiles_df", '*.{}'.format('parquet')))
    for f in parquet_files:
        df = pd.read_parquet(f).fillna(0)/1000
        # df.columns = df.columns.droplevel(level = 1)
        loads_standard_profiles[f.split(os.sep)[-1][:-8]]  = df.set_index(pd.date_range(start="00:00", freq="1s", periods = len(df.index))).resample(time_step).mean()
        # fig, ax = plt.subplots(figsize = (10,10))
        # loads_standard_profiles[f.split(os.sep)[-1][:-8]].plot(ax = ax)
        # fig.savefig(os.path.join("synelprof","plots", f'{f.split(os.sep)[-1][:-8]}.png'))
        # plt.close()
    return loads_standard_profiles

def get_length_variable_appliance(appliances):
    v = {
        "Electric cooking": [120, 120, 1800],
        "Electric oven": [120, 300, 2400],
        "Television": [120, 3600 * 1.5, 3600 * 6],
        "Monitor": [120, 3600, 3600 * 5],
        "Light": [120, 3600, 3600 * 6],
    }[appliances] # Values from
    minimum = v[0]
    mode = v[1]
    maximum = v[2]

    d = (minimum + 4*mode + maximum)/6
    al = 6*((d - minimum)/(maximum - minimum))
    be = 6*((maximum - d)/(maximum - minimum))
    loc = minimum
    scale = maximum - minimum

    random_values = np.random.randint(low=0, high=maximum, size=10000)
    x = np.linspace(minimum,maximum, 100)
    return beta(al, be, loc, scale)

def generation_profile(consumo_annuale,
                       provincia,
                       giorno_iniziale,
                       mercato = "Tutti",
                       fp = "FP3",
                       residenza = 'Tutti'):

    pass


def create_synthetic_profile(
        n_dwelling,
        province,
        region,
        timestep_per_hour = 1,
        power_range = "FP3",
        initial_day = 0,
):
    numero_utenze = n_dwelling
    provincia = province
    region = region
    mercato = "Tutti"
    fp = power_range
    residenza = "Tutti"
    giorno_iniziale = initial_day # Wednesday
    time_step = f"{int(60/timestep_per_hour)}min"
    ts_per_hour = timestep_per_hour

    # consumo_annuale = 3000. # kWh
    # holidays = [1,2,3,180,181,182,364,365]

    loads = get_italian_random_el_consumption(numero_utenze, region)
    distribution_ = average_profile_creation_from_ARERA(provincia,giorno_iniziale,time_step,mercato="Tutti",fp="FP3",residenza='Tutti')
    distribution = distribution_ / distribution_.sum(axis = 0)
    loads_standard_profiles = get_standard_app_profiles(time_step)


    total_cons = np.zeros([24*ts_per_hour*365, numero_utenze])
    el_load_matrixes = {}
    start = time.time()
    for dw in range(numero_utenze):
        el_load_matrixes[dw] = {}
        # total_cons[dw] = np.zeros([24*12*365])
        for load in loads.columns:
            # First category
            if load in ['Refrigerator', 'Freezer']:
                app_key = {
                    "Refrigerator": "Fridge",
                    "Freezer": "Freezer"
                }[load]
                # Freezer, Fridge
                el_load = np.zeros([24*ts_per_hour,365])
                for i in range(365):
                    profile = loads_standard_profiles[app_key].sample(axis='columns').iloc(axis=1)[0].values
                    el_load[:,i] = np.roll(profile, np.random.randint(0,len(profile)))

                if np.sum(el_load) > 0:
                    el_load_matrixes[dw][load] = el_load/(np.sum(el_load)/ts_per_hour) * loads.loc[dw][load]
                else:
                    el_load_matrixes[dw][load] = el_load
                total_cons[:,dw] += el_load_matrixes[dw][load].T.reshape(8760*ts_per_hour)

            if load in ['Washing machine', 'Clothes dryer','Dish washer',]:
                app_key = {
                    "Washing machine": "Washing_machine",
                    "Clothes dryer": "Tumble_dryer",
                    "Dish washer": "Dishwasher"
                }[load]
                load_profile = loads_standard_profiles[app_key].sample(axis='columns').iloc(axis=1)[0].values
                sample = np.random.rand(365)
                number_of_daily_on_ev_int = (loads.loc[dw][load] / 365 / (load_profile.sum()/ts_per_hour))//1
                if number_of_daily_on_ev_int < 0:
                    number_of_daily_on_ev_int = 0
                number_of_daily_on_ev = sample < (loads.loc[dw][load] / 365 / (load_profile.sum()/ts_per_hour))%1
                number_of_daily_on_ev = (number_of_daily_on_ev + number_of_daily_on_ev_int).astype(int)
                y_guess = np.random.rand(np.max(number_of_daily_on_ev),365)
                cdf = np.cumsum(distribution, axis = 0)
                x_event = np.array([np.interp(y_guess[:,i], cdf[:,i], np.arange(len(cdf[:,0]))) for i in range(365)]).astype(int)

                load_profile_time_step_average = int(np.interp(np.cumsum(load_profile).mean(), np.cumsum(load_profile), np.arange(len(load_profile))))
                el_load = np.zeros([3*24*ts_per_hour, 365])
                x_event = x_event - load_profile_time_step_average + 24 * ts_per_hour
                x_end = x_event + len(load_profile)

                for d in np.arange(365)[number_of_daily_on_ev.astype(bool)]:
                    n_acc = number_of_daily_on_ev[d]
                    for n_a in range(n_acc):
                        el_load[x_event[d,n_a]:x_end[d,n_a], d] += load_profile

                el_load[24*ts_per_hour:24*ts_per_hour*2,:-1] += el_load[:24*ts_per_hour,1:]
                el_load[24*ts_per_hour:24*ts_per_hour*2,1:] += el_load[24*ts_per_hour*2:24*ts_per_hour*3,:-1]
                el_load = el_load[24*ts_per_hour:24*ts_per_hour*2,:]
                if np.sum(el_load) > 0:
                    el_load_matrixes[dw][load] = el_load/(np.sum(el_load)/ts_per_hour) * loads.loc[dw][load]
                else:
                    el_load_matrixes[dw][load] = el_load

                total_cons[:,dw] += el_load_matrixes[dw][load].T.reshape(8760*ts_per_hour)

            if load in ['Television','Electric cooking', 'Electric oven', 'Monitor', 'Light',]:
                app_key = {
                    "Electric cooking": "Electric_hobs",
                    "Electric oven": "Electric_oven",
                    "Television": "TV",
                    "Monitor": "PC",
                    "Light": "Lights",
                }[load]
                distribution_length = get_length_variable_appliance(load)
                daily_length_ts = (distribution_length.rvs(size=365) / 3600 * ts_per_hour).astype(int)
                daily_length_ts[daily_length_ts==0] = 1
                load_profile_tot = loads_standard_profiles[app_key].sample(axis='columns').iloc(axis=1)[0].values
                sample = np.random.rand(365)

                min_daily_length = np.min(daily_length_ts)
                max_number_of_daily_on_ev_int = int(np.round(loads.loc[dw][load] / 365 / (load_profile_tot[:int(min_daily_length)].sum() / ts_per_hour)))
                y_guess = np.random.rand(np.max(max_number_of_daily_on_ev_int), 365)
                cdf = np.cumsum(distribution, axis=0)
                x_event = np.array(
                    [np.interp(y_guess[:, i], cdf[:, i], np.arange(len(cdf[:, 0]))) for i in range(365)]).astype(int)
                x_end = np.zeros(x_event.shape).astype(int)
                el_load = np.zeros([3*24*ts_per_hour, 365])
                for d in range(365):
                    load_profile = load_profile_tot[:int(daily_length_ts[d])]
                    load_profile_time_step_average = int(np.interp(np.cumsum(load_profile).mean(), np.cumsum(load_profile), np.arange(len(load_profile))))
                    x_event[d,:] = x_event[d,:] - load_profile_time_step_average + 24 * ts_per_hour
                    x_end[d,:] = x_event[d,:] + len(load_profile)

                    number_of_daily_on_ev = int(np.round(loads.loc[dw][load] / 365 / (load_profile.sum()/ts_per_hour)))
                    if number_of_daily_on_ev > 0:
                        n_acc = number_of_daily_on_ev
                        for n_a in range(n_acc):
                            el_load[x_event[d, n_a]:x_end[d, n_a], d] += load_profile

                el_load[24*ts_per_hour:24*ts_per_hour*2,:-1] += el_load[:24*ts_per_hour,1:]
                el_load[24*ts_per_hour:24*ts_per_hour*2,1:] += el_load[24*ts_per_hour*2:24*ts_per_hour*3,:-1]
                el_load = el_load[24*ts_per_hour:24*ts_per_hour*2,:]
                if np.sum(el_load) > 0:
                    el_load_matrixes[dw][load] = el_load/(np.sum(el_load)/ts_per_hour) * loads.loc[dw][load]
                else:
                    el_load_matrixes[dw][load] = el_load
                total_cons[:,dw] += el_load_matrixes[dw][load].T.reshape(8760*ts_per_hour)

            if load in ['Small appliances']:
                # 
                pass

    return total_cons, loads
    # np.savetxt("ts_results.csv",total_cons, delimiter = ";")
    # loads.to_csv("annual_results.csv")
    #
    # stop = time.time()
    # print(f"""
    # Simulation time: {stop-start:.2f} s
    # Simulation per dw: {(stop-start)/numero_utenze:.2f} s
    # """)
    # plt.style.use('ggplot')
    # plt.rcParams['figure.facecolor'] = "#E9E9E9"
    # plt.rcParams['axes.facecolor'] = "white"
    # plt.rcParams['grid.color'] = "#E5E5E5"
    # plt.rcParams['xtick.color'] = "black"
    # plt.rcParams['ytick.color'] = "black"
    # plt.rcParams['axes.edgecolor'] = "black"
    # plt.rcParams["legend.edgecolor"] = 'black'
    # plt.rcParams["legend.facecolor"] = 'white'
    # plt.rcParams["xtick.labelcolor"] = 'black'
    # plt.rcParams["ytick.labelcolor"] = 'black'


    # fig, [ax1,ax2] = plt.subplots(nrows = 2, figsize = (10,8))
    # ax1.plot(total_cons[:,:10*24*60])
    # ax1.set_xlabel("Time [min]")
    # ax1.set_ylabel("Electric power [W]")
    # ax1.ticklabel_format(axis='Y',style = 'scientific')
    # ax1.set_ylim(0,5)
    #
    # ax2.plot(total_cons.mean(axis = 1).reshape(365,ts_per_hour*24).mean(axis = 0))
    # ax2.set_xlabel("Time [min]")
    # ax2.set_ylabel("Electric power [W]")
    # ax2.ticklabel_format(axis='Y',style = 'scientific')
    #
    # plt.tight_layout()
    # plt.show()
    #
    #
    # #
    #
    # fig, ax1 = plt.subplots(ncols = 1, figsize = (6,4))
    # ax1.plot(distribution_[:,2:5], label = ["Weekday","Satuday","Sunday"])
    # ax1.legend()
    # ax1.set_xlabel("Time [min]")
    # ax1.set_ylabel("ToU PDF [-]")
    # ax1.ticklabel_format(axis='Y',style = 'scientific')
    # # ax1.set_ylim(0,0.0012)
    # plt.tight_layout()
    # plt.show()

    # ax2.plot(total_cons)




    # fig, ax = plt.subplots()
    # pd.Series(distribution.T.reshape(8760*12), index = pd.date_range("01/01/2023 00:00", periods = 8760*12, freq = time_step)).plot(ax = ax, marker = 'o')
    #
    # for m in pd.date_range("01/01/2023 23:59", periods = 12, freq = "1M"):
    #     ax.axvline(x = m, color = 'r')
