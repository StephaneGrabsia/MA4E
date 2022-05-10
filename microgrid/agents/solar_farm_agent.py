import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
import numpy as np
import pulp

class SolarFarmAgent:
    def __init__(self, env: SolarFarmEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        a = self.env.action_space.sample()
        a = (np.tanh(state['manager_signal'])) / 10
        if state['soc'] + a[0] * 0.5 > self.env.battery.capacity:
            a[0] = (self.env.battery.capacity - state['soc']) * 2
        if state['soc'] + a[0] * 0.5 < 0:
            a[0] = - state['soc'] * 2
        return a


    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        pb = pulp.LpProblem("ferme_solaire", pulp.LpMinimize)
        # variables
        conso_totale = {}
        conso_batterie = {}
        conso_batterie_pos = {}
        conso_batterie_neg = {}
        signe_conso_batterie = {}
        prod_PV = {}
        charge_batterie = {}
        prix = {}

        # constantes
        T = env.nb_pdt
        dt = env.delta_t

        charge_batterie[0] = env.battery.soc #state[soc]
        rendement_charge = env.battery.efficiency
        rendement_decharge = env.battery.efficiency
        charge_max = env.battery.capacity
        puissance_max = env.battery.pmax
        surface = env.pv.surface

        # on peut exploiter les données du manager et de l'environnement en terme de prévision
        prod_PV = 0.001 * surface * state['pv_prevision']
        prix = state['manager_signal']

        for t in range(T):
            conso_totale[t] = pulp.LpVariable("conso_totale_" + str(t), None, None)
            conso_batterie[t] = pulp.LpVariable("conso_batterie_" + str(t), -puissance_max, puissance_max)
            conso_batterie_pos[t] = pulp.LpVariable("conso_batterie_pos_" + str(t), 0, puissance_max)
            conso_batterie_neg[t] = pulp.LpVariable("conso_batterie_neg_" + str(t), 0, puissance_max)
            signe_conso_batterie[t] = pulp.LpVariable("signe_conso_batterie_" + str(t), cat=pulp.LpBinary)
            charge_batterie[t + 1] = pulp.LpVariable("charge_batterie_" + str(t), 0, charge_max)

            # creation des contraintes
            pb += conso_batterie[t] == conso_batterie_pos[t] - conso_batterie_neg[t], "conso_batterie_" + str(t)
            pb += conso_batterie_pos[t] <= signe_conso_batterie[t] * puissance_max, "conso_pos_" + str(t)
            pb += conso_batterie_neg[t] <= (1 - signe_conso_batterie[t]) * puissance_max, "conso_neg_" + str(t)
            pb += conso_totale[t] == conso_batterie[t] - prod_PV[t], "conso_tot_" + str(t)
            pb += charge_batterie[t + 1] == charge_batterie[t] + (
                        rendement_charge * conso_batterie_pos[t] - conso_batterie_neg[t] * (
                            1 / rendement_decharge)) * dt, "charge_batterie_" + str(t)

        # creation de la fonction objectif
        pb.setObjective(pulp.lpSum([conso_totale[t] * prix[t]] for t in range(T)) * dt)

        pb.solve()
        a = self.env.action_space.sample()
        for t in range(T):
            a[t]=conso_totale[t]

        return a


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    solar_farm_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'pv': {
            'surface': 100,
            'location': "enpc",  # or (lat, long) in float
            'tilt': 30,  # in degree
            'azimuth': 180,  # in degree from North
            'tracking': None,  # None, 'horizontal', 'dual'
        }
    }
    env = SolarFarmEnv(solar_farm_config=solar_farm_config, nb_pdt=N)
    agent = SolarFarmAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))