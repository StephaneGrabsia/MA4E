
import sys
sys.path += ["C:\Dossiers\optim_et_energie\Git\MA4E"]

import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv
import numpy as np
import pulp as pl


class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        
        #données du manager
        #consumption_prevision = state.get("consumption_prevision") #la demande de consommation
        consumption_prevision = state['consumption_prevision']
        #soc = state.get("soc") #stock batterie
        soc = state['soc']
        #manager_signal = state.get("manager_signal") #les prix
        manager_signal = state['manager_signal']
        #date_time = state.get("datetime")
        date_time = state['datetime']
        
        #constantes batterie
        pmax = self.env.battery.pmax #puissance maximale de la batterie
        efficiency = self.env.battery.efficiency #rendement batterie, mâme pour charge/decharge
        capacity = self.env.battery.capacity #charge maximale
        a_tdec = self.env.battery.soc #state[soc]

        #constantes temps
        delta_t = datetime.timedelta(minutes=30)
        H = datetime.timedelta(hours=1)
        T = self.env.nb_pdt #Nombre de périodes temporelles
        liste_temps = [t for t in range(T)]

        #création du problème
        prob = pl.LpProblem("industrial_site", pl.LpMinimize)

        #définition des variables
        a = pl.LpVariable.dicts("batterie_stock",liste_temps,0,capacity)
        l_bat_plus = pl.LpVariable.dicts("l_batterie_plus",liste_temps,0)
        l_bat_moins = pl.LpVariable.dicts("l_batterie_moins",liste_temps,0)
        l_bat = pl.LpVariable.dicts("l_batterie",liste_temps)
        l_tot = pl.LpVariable.dicts("l_demande_totale",liste_temps)

        #Fonction objectif
        
        prob += pl.lpSum([l_tot[t] * manager_signal[t] * delta_t/H for t in liste_temps])

        #contraintes

        #prob += a[0] == a_tdec
        
        for t in range(T):

            prob += l_bat_plus[t] + l_bat_moins[t] <= pmax
            prob += l_bat[t] == l_bat_plus[t] - l_bat_moins[t]
            prob += l_tot[t] == consumption_prevision[t] + l_bat[t]
            if t>0:
                prob += a[t] == a[t-1] +(efficiency*l_bat_plus[t] - l_bat_moins[t]*1/efficiency)*delta_t/H
            else :
                prob += a[t] == a_tdec + (efficiency*l_bat_plus[t] - l_bat_moins[t]*1/efficiency)*delta_t/H

        prob.solve()
        result = self.env.action_space.sample()
        for t in range(T):
            result[t] = l_bat[t].value()
        return result




if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(2*N):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        if reward <0:
            break

