import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from synelprof.functions import create_synthetic_profile

for i in range(100): 
    print(i)
    path = os.path.join("results_Padova")
    if not os.path.isdir(path):
        os.mkdir(path)
    total_cons, loads, el_load_matrixes, counters = create_synthetic_profile(
        n_dwelling = 100,
        province = "Padova",
        region = "Veneto",
        timestep_per_hour=2,
        power_range="FP2",
        initial_day=0,
        run_syn_prof = True,
    )

    np.savetxt(os.path.join(path,f"ts_results_FP2_{i}.csv"),total_cons, delimiter = ";")
    loads.to_csv(os.path.join(path,f"annual_results_FP2_{i}.csv"), sep = ";")
    
    total_cons, loads, el_load_matrixes, counters = create_synthetic_profile(
        n_dwelling = 100,
        province = "Padova",
        region = "Veneto",
        timestep_per_hour=2,
        power_range="FP3",
        initial_day=0,
        run_syn_prof = True,
    )

    np.savetxt(os.path.join(path,f"ts_results_FP3_{i}.csv"),total_cons, delimiter = ";")
    loads.to_csv(os.path.join(path,f"annual_results_FP3_{i}.csv"), sep = ";")
    
    total_cons, loads, el_load_matrixes, counters = create_synthetic_profile(
        n_dwelling = 100,
        province = "Padova",
        region = "Veneto",
        timestep_per_hour=2,
        power_range="FP4",
        initial_day=0,
        run_syn_prof = True,
    )

    np.savetxt(os.path.join(path,f"ts_results_FP4_{i}.csv"),total_cons, delimiter = ";")
    loads.to_csv(os.path.join(path,f"annual_results_FP4_{i}.csv"), sep = ";")

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = "#E9E9E9"
plt.rcParams['axes.facecolor'] = "white"
plt.rcParams['grid.color'] = "#E5E5E5"
plt.rcParams['xtick.color'] = "black"
plt.rcParams['ytick.color'] = "black"
plt.rcParams['axes.edgecolor'] = "black"
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["xtick.labelcolor"] = 'black'
plt.rcParams["ytick.labelcolor"] = 'black'


fig, [ax1,ax2] = plt.subplots(nrows = 2, figsize = (10,8))
ax1.plot(total_cons[:,:10*24*60])
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Electric power [W]")
ax1.ticklabel_format(axis='Y',style = 'scientific')
ax1.set_ylim(0,5)

ax2.plot(total_cons.mean(axis = 1).reshape(365,1*24).mean(axis = 0))
ax2.set_xlabel("Time [min]")
ax2.set_ylabel("Electric power [W]")
ax2.ticklabel_format(axis='Y',style = 'scientific')

plt.tight_layout()
plt.show()
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