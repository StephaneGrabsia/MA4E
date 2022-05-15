#import sys
#sys.path += ["D:/ENPC/1A/Cours/COUV/Optimisation et énergie/git_microgrid/MA4E"]

import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv



class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):

        l_it = state['consumption_prevision']
        lbd = state["manager_signal"]
        p_hw = state["hotwater_price_prevision"]

        res = self.env.action_space.sample()

        K1 = ( self.env.COP_HP * self.env.COP_CS ) / (self.env.EER * (self.env.COP_HP - 1) )
        K2 = self.env.COP_CS / (self.env.EER * (self.env.COP_HP - 1) * (self.delta_t/datetime.timedelta(hours=1) ) )
        
        for t in range(0,self.env.nb_pdt):
        
            max_alpha = self.data_center.get_max_alpha_t(self.now + t*self.delta_t , self.delta_t)

            if (l_it[t] > 0 ): # si la demande en énergie du centre informatique n'est pas nulle à cet instant
            
                if (K1 * p_hw[t] > K2 * lbd[t] ) : # ie si on est bénéficiaire à envoyer de l'énergie thermique
                    res[t] = max_alpha  # on en envoie la proportion maximale

                else:
                    res[t] = 0 # sinon on en envoie pas du tout

            else: # si la demande en énergie est nulle on considère que l'on ne renvoie aucune énergie
                res[t] = 0
            
        return res


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    data_center_config = {
        'scenario': 10,
    }
    env = DataCenterEnv(data_center_config, nb_pdt=N)
    agent = DataCenterAgent(env)
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