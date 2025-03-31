import matplotlib

matplotlib.use('TkAgg',force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("bmh")

from synelprof.electric_load_italian_distribution import electric_load_italian_dict

plt.rcParams['axes.facecolor'] = "white"

labels_dev = pd.read_excel("table.xlsx", sheet_name ="Sheet1", header = 0, index_col = 0)

rows = [f[4:] for f in electric_load_italian_dict.keys() if "penetration" not in f]
data = pd.DataFrame(index = rows,columns = ["PDF mean", "PDF std",
                                            "PDF min",
                                            "PDF max",
                                            "PDF 25",
                                            "PDF 75"])


data_samp = pd.DataFrame()

for l in electric_load_italian_dict.keys():
    if "penetration" not in l:


        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax_ = ax.twinx()
        # x = np.linspace(0, 1000, 100)

        max_lim = 0.001
        if l != 'Tot':
            samp = electric_load_italian_dict[f"{l}"].rvs(size=100000)

            data_samp[l[4:]] = samp

            data.loc[l[4:],"PDF mean"] = np.mean(samp)
            data.loc[l[4:],"PDF std"] = np.std(samp)
            # data["PDF min"].loc[l[4:]] = np.quantile(samp, 0.)
            # data["PDF max"].loc[l[4:]]= np.quantile(samp, 1.)
            data.loc[l[4:],"PDF 25"] = np.quantile(samp, 0.25)
            data.loc[l[4:],"PDF 75"] = np.quantile(samp, .75)
            # ax_.plot(x, electric_load_italian_dict[f"{l}"].pdf(x), lw=2, alpha=0.6, label=f'{l}', color='r')
            # max_lim = np.max(electric_load_italian_dict[f"{l}"].pdf(x))
        # ax_.set_ylim(0, max_lim)
        # ax.set_ylim(0, max_lim)
        # plt.tight_layout()
        # plt.show()
        # plt.close()

data = data.rename({
    "Dish washer": "Dishwasher",
    "Television": "TV and PC",
})
data = data.drop([
    "Electric cooking",
    "Monitor",
    "Small appliances",
    "Cooling split"
])
data_samp = data_samp.rename({
    "Dish washer": "Dishwasher",
    "Television": "TV and PC",
}, axis = 1)
data_samp = data_samp.drop([
    "Electric cooking",
    "Monitor",
    "Small appliances",
    "Cooling split"
], axis = 1)

means = data["PDF mean"].values
q1 = data["PDF 25"].values
q3 = data["PDF 75"].values
whislo = data["PDF min"].values
whishi = data["PDF max"].values

keys = ['med', 'q1', 'q3', 'whislo', 'whishi']
stats = [dict(zip(keys, vals)) for vals in zip(means, q1, q3, whislo, whishi)]

fig, ax = plt.subplots(figsize = (5,5))

# ax.bxp(stats, showfliers=False, label = "Model distribution")

ax.violinplot(data_samp,vert=True, widths=0.5,
              showmeans=False, showextrema=False, showmedians=False)
ax.set_xticks([i for i in range(1,9)])
ax.set_xticklabels(data_samp.columns, rotation=45, rotation_mode="anchor", ha='right')
ax.plot()

i = 1
for dev in data.index:
    ax.scatter(x = i, y = labels_dev["Lower Limit [kWh/year]"][dev], marker = 'o', color = 'blue' )
    ax.scatter(x=i, y=labels_dev["Upper Limit [kWh/year]"][dev], marker='^', color='red')

    i+=1

i-=1
ax.scatter(x = i, y = labels_dev["Lower Limit [kWh/year]"][dev], marker = 'o', color = 'blue', label = "Best energy label")
ax.scatter(x=i, y=labels_dev["Upper Limit [kWh/year]"][dev], marker='^', color='red', label = "Worst energy label")

ax.set_ylabel("Annual electric consumption [kWh]")
ax.legend(loc = "upper left")

ax.set_ylim([0, 850])

plt.tight_layout()
fig.savefig("plot_for_paper.svg")
fig.savefig("plot_for_paper.png", dpi = 500)
plt.show(block=True)
