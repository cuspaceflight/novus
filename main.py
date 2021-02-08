"""Nitrous oxide hybrid rocket burn simulator."""

########################################
# Joe Hunt updated 13/12/17            #
# All units SI unless otherwise stated #
########################################
# Prettified by Henry Franks, 8/2/21   #
########################################

import numpy as np
import matplotlib.pyplot as plt
from n2o_esdu import thermophys


ZERO_C = 273.15

# inputs
Vtank = 53e-3           # tank volume (litres)
head = 0.07             # initial vapour phase proportion
Ninj = 52               # number of injector orifices
Dinj = 0.0013           # diameter of injector orifices
Dport = 0.0789          # diameter of fuel port
Dfuel = 0.112           # Outside diameter of fuel grain
Lport = 1.33            # length of fuel port
Dthroat = 0.0438        # nozzle throat diameter
noz_cone_angle = 15     # nozzle exit cone angle (degrees)
temp = 5 + ZERO_C       # initial tank temperature
fuelden = 935           # solid fuel density
Dfeed = 0.023           # minimum feed system orifice diameter
reg_coeff = 'blank'     # regression rate coeff (usually 'a' in textbooks)
reg_exp = 'blank'       # regression rate exponent (usually 'n' in textbooks)
external_pres = 101325  # external atmospheric pressure
dt = 0.001              # time step


def A_from_D(D):
    """Get pipe area from diameter."""
    return np.pi * pow(D, 2) / 4


def main():
    """Run simulation."""
    with open('L_Nitrous_S_HDPE.propep', 'r') as propep:
        # stream propep data file
        propep_data = propep.readlines()

        # from datafile, create lists of O/F and
        # characteristic velocity for interpolation
        OFdat2, Cstardat2 = [], []
        for n in reversed(range(191)):
            OFdat2.append(float(propep_data[4 * n + 9].split()[1]))
            Cstarfps = float(propep_data[4 * n + 11].split()[4])
            Cstardat2.append(Cstarfps * 0.3048)

        # check input temp and regression constants are in range
        if not (183.15 <= temp <= 309.57):
            raise ValueError('Input ambient temperature out of data range')

        if type(reg_coeff) == str or type(reg_exp) == str:
            raise ValueError('Regression rate constant input(s) blank')

        # assign initial values
        K = 2
        vapzl = 0
        time = 0
        mdotox = 0
        nopulse = True
        notransient = True
        Impulse = 0
        (timelist, vpreslist, cpreslist, Thrustlist,
        Isplist, Goxlist, Dportlist, manifoldpreslist) = ([], [], [], [],
                                                        [], [], [], [])
        lden, vden, hv, c, vpres = thermophys(temp)
        ilmass = Vtank * (1 - head) * lden
        lmass = ilmass
        ivmass = Vtank * head * vden
        vmass = ivmass
        ifuelmass = (A_from_D(Dfuel) - A_from_D(Dport)) * Lport * fuelden
        fuelmass = ifuelmass
        tmass = lmass + vmass
        cpres = external_pres

        print("Initial conditions:\ntime:", time,
            "s\ntemp:", temp - ZERO_C,
            "C\nlmass:", lmass,
            "kg\nvmass:", vmass,
            "kg\nvpres:", vpres,
            'Pa\nfuel thickness:', (Dfuel - Dport) / 2,
            'm\nfuel mass', fuelmass, 'kg\n')

        # sim loop
        while lmass > 0 and Dport < Dfuel:
            time += dt

            # update nitrous thermophysical properties if temperature in range
            temp -= (vapzl * hv) / (lmass * c)
            if not (183.15 <= temp <= 309.57):
                raise RuntimeError('temperature left data range')
            lden, vden, hv, c, vpres = thermophys(temp)

            # calculate injector pressure drop
            feed_pdrop = (0.5 * lden * pow(mdotox / (lden * A_from_D(Dfeed)), 2))
            manifoldpres = vpres - feed_pdrop
            inj_pdrop = manifoldpres - cpres

            if inj_pdrop < 0 and time > 0.5:
                print('Motor exploded ;_; Reverse flow occurred at t=', time, 's')
                break
            if inj_pdrop < 0 and time < 0.5:
                print('Warning! Chamber pressure exceeded '
                    'tank pressure during ignition.')
                inj_pdrop = 0
                notransient = False
            if (inj_pdrop / cpres) < 0.2 and nopulse and time > 0.5:
                print("PULSE began at T=", time, "s")
                nopulse = False

            # injector flow-rate calculation
            nmdotox = pow(2 * lden * inj_pdrop /
                        (K / pow(Ninj * A_from_D(Dinj), 2)), 0.5)
            mdotox = (mdotox + (3 * nmdotox)) / 4

            # nitrous vaporization calculations
            tmass = tmass - (mdotox * dt)
            lmass_pre_vap = lmass - (mdotox * dt)
            lmass = (Vtank - (tmass / vden)) / ((1 / lden) - (1 / vden))

            if (lmass_pre_vap < lmass):
                print('loop exited early due to numerical instability')
                break

            vapz = lmass_pre_vap - lmass
            vapzl = (dt / 0.15) * (vapz - vapzl) + vapzl
            vmass = tmass - lmass

            # fuel port calculations
            if mdotox / A_from_D(Dport) > 600 and time > 0.5:
                print('Ignition failure: oxidizer flux too high...')
                break

            rdot = reg_coeff * pow(mdotox / A_from_D(Dport), reg_exp)

            mdotfuel = rdot * fuelden * (np.pi * Dport * Lport)
            OF = mdotox/mdotfuel

            if not (1/39 <= OF <= 39):
                raise ValueError('OF out of propep data range!')

            Cstar = np.interp((OF / (OF + 1)) * 100, OFdat2, Cstardat2)
            cpres = (mdotox + mdotfuel) * Cstar / A_from_D(Dthroat)
            Dport += 2 * rdot * dt
            fuelmass = (A_from_D(Dfuel) - A_from_D(Dport)) * Lport * fuelden

            # lookup ratio of specific heats from propep data file
            if cpres > 90E5 or cpres < 0:
                raise RuntimeError('chamber pressure out of propep data range!')

            rounded_cpres = int(5 * round((cpres * 1e-5) / 5))
            cpres_line = int(765 * (((100 - rounded_cpres) / 5) - 1) + 9)
            rounded_oxpct = round(2 * (OF / (1 + OF)) * 100) / 2
            oxpct_line = int((4 * (97.5 - rounded_oxpct) / 0.5) + 3)
            γ = float(propep_data[cpres_line + oxpct_line - 1].split()[1])

            # performance calculations
            Cf = pow(γ
                    * pow(2 / (γ + 1), (γ + 1) / (γ - 1))
                    * (2 * γ) / (γ - 1)
                    * (1 - pow(external_pres / cpres, (γ - 1) / γ)), 0.5)

            Thrust = (A_from_D(Dthroat) * cpres * Cf *
                    0.5 * (1 + np.cos(2 * np.pi * noz_cone_angle / 360)))

            Isp = Thrust / ((mdotox + mdotfuel) * 9.81)
            Impulse = Impulse + (Thrust * dt)

            # add current state to plot lists
            timelist.append(time)
            vpreslist.append(vpres)
            cpreslist.append(cpres)
            manifoldpreslist.append(manifoldpres)
            Thrustlist.append(Thrust)
            Isplist.append(Isp)
            Goxlist.append(mdotox / (((Dport/2)**2)*np.pi))
            Dportlist.append(Dport)

        # print final results
        print("\nFinal conditions:\ntime:", time,
            "s\ntemp:", temp - ZERO_C,
            "C\nlmass:", lmass,
            "kg\nvmass:", vmass,
            "kg\nvpres:", vpres,
            'Pa\nfuel thickness:', (Dfuel - Dport)/2,
            'm\nfuel mass', fuelmass, 'kg')

        print('\nPerformance results:\nmean thrust:', np.mean(Thrustlist),
            'N\nimpulse:', Impulse,
            'Ns\nmean Isp:', np.mean(Isplist),
            's\noxidizer burnt:', ilmass - (vmass - ivmass),
            'kg\nfuel burnt', ifuelmass - fuelmass, 'kg')

        print('\ntrajectory sim input:\nithrust =', Thrustlist[int(0.5/dt)],
            '\nthrustgrad =', (Thrustlist[int(0.5 / dt)] - Thrust)/time,
            '\nburntime =', time,
            '\npropmass =', (ilmass - (vmass - ivmass))+(ifuelmass - fuelmass),
            '\nvapmass =', vmass)

        # plot pressures
        plt.figure(figsize=(8.5, 7))
        plt.subplot(221)
        plt.plot(timelist, vpreslist, 'b', label='Tank pressure')
        plt.plot(timelist, cpreslist, 'r', label='Chamber pressure')
        plt.plot(timelist, manifoldpreslist, 'g',
                label='Injector manifold pressure')
        plt.ylabel('Pressure (Pa)')
        plt.ylim(0, max(vpreslist)*1.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (Pa)')
        plt.legend()
        plt.tight_layout()

        # plot thrust
        plt.subplot(222)
        plt.plot(timelist, Thrustlist)
        plt.xlabel('Time (s)')
        plt.ylabel('Thrust (N)')
        plt.ylim(0, max(Thrustlist)*1.3)
        plt.tight_layout()

        # plot massflux
        plt.subplot(223)
        plt.plot(timelist, Goxlist, 'y')
        plt.xlabel('Time (s)')
        plt.ylabel('Oxidizer mass flux ($kg s^{-1} m^{-2}$)')
        plt.ylim(0, max(Goxlist)*1.3)
        plt.tight_layout()

        # plot port diameter
        plt.subplot(224)
        plt.plot(timelist, Dportlist, 'g')
        plt.xlabel('Time (s)')
        plt.ylabel('Port Diameter (m)')
        plt.ylim(0, max(Dportlist)*1.3)
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()
