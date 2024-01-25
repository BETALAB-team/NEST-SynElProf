import numpy as np
import matplotlib.pyplot as plt

import richardsonpy.classes.occupancy as occ
import richardsonpy.functions.change_resolution as cr
import richardsonpy.functions.load_radiation as loadrad
import richardsonpy.classes.electric_load as eload

#  Total number of occupants in apartment
nb_occ = 3

timestep = 600  # in seconds

#  Generate occupancy object (necessary as input for electric load gen.)

#  Get radiation (necessary for lighting usage calculation)
(q_direct, q_diffuse) = loadrad.get_rad_from_try_path()

#  Convert 3600 s timestep to given timestep
q_direct = cr.change_resolution(q_direct, old_res=3600, new_res=timestep)
q_diffuse = cr.change_resolution(q_diffuse, old_res=3600, new_res=timestep)

#  Generate stochastic electric load object instance
el_load = []
for i in range(10):
    print(i)
    occ_obj = occ.Occupancy(number_occupants=nb_occ)
    el_load_obj = eload.ElectricLoad(occ_profile=occ_obj.occupancy,
                                     total_nb_occ=nb_occ,
                                     q_direct=q_direct,
                                     q_diffuse=q_diffuse,
                                     timestep=timestep)

    #  Calculate el. energy in kWh by accessing loadcurve attribute
    energy_el_kwh = sum(el_load_obj.loadcurve) * timestep / (3600 * 1000)

    print('Electric energy demand in kWh: ')
    print(energy_el_kwh)
    el_load.append(el_load_obj.loadcurve)

el_load = np.array(el_load).T

plt.plot(occ_obj.occupancy)
plt.plot()