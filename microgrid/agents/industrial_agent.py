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
        consumption_prevision = state.get("consumption_prevision") #la demande de consommation
        soc = state.get("soc") #à utiliser au 2nd run, pas aujourd'hui
        manager_signal = state.get("manager_signal") #les prix
        datetime = state.get("datetime") #à utiliser au 2nd run, pas aujourd'hui
        #H = datetime.timedelta(hours=1)
        pmax = self.env.battery.pmax #utilisé
        efficiency = self.env.battery.efficiency #utilisé
        capacity = self.env.battery.capacity #utilisé


        #initialisation

        a_tdec = 0 #au second run, voir comment récupérer le vrai stock au temps 0 de la simulation
        T = [t for t in range(self.nb_pdt)]

        #création du problème

        prob = pl.LpProblem("industrial_site", pl.LpMinimize)

        #définition des variables

        a = pl.LpVariable.dicts("batterie_stock",T,0,capacity)
        l_bat_plus = pl.LpVariable.dicts("l_batterie_plus",T,0)
        l_bat_moins = pl.LpVariable.dicts("l_batterie_moins",T,0)
        l_bat = pl.LpVariable.dicts("l_batterie",T)
        l_tot = pl.LpVariable.dicts("l_demande_totale",T)

        "Fonction objectif"
        
        prob += pl.lpSum([l_tot[t] * manager_signal[t] * delta_t for t in T])

        #contraintes

        prob += a[0] == a_tdec
        
        for t in range(self.nb_pdt):

            prob += l_bat_plus[t] + l_bat_moins[t] <= pmax
            prob += l_bat[t] == l_bat_plus[t] - l_bat_moins[t] #pas besoin de l_bat en variable pour la prochaine fois
            prob += l_tot[t] == consumption_prevision[t] + l_bat[t] #pas besoin de l_tot en variable pour la prochaine fois
            if t>0:
                prob += a[t] == a[t-1] +(efficiency*l_bat_plus[t] - l_bat_moins[t]*1/efficiency)*delta_t


        prob.solve()


        results = [0] * self.nb_pdt
        for t in range(self.nb_pdt):
            results[t] = l_bat[t].value()

        return np.array(results)




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
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
