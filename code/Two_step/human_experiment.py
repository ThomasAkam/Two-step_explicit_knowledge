 # Class containing data from all sessions in one experiment.

import os
import pickle
import numpy as np
from . import human_session as hs
from . import plotting as pl

class experiment:
    def __init__(self, exp_name, rebuild_sessions=False):
        '''
        Instantiate an experiment object for specified group number.  Tries to load previously 
        saved sessions,  then loads sessions from data folder if they were not in
        the list of loaded sessions and are from animals in the group.  rebuild sessions argument
        forces the sessions to be created directly from the data files rather than loaded.
        '''

        self.name = exp_name
        self.path = os.path.join('..', 'data', exp_name)

        self.sessions = []
        
        if not rebuild_sessions:
            try:
                exp_file = open(os.path.join(self.path, 'sessions.pkl'),'rb')
                self.sessions = pickle.load(exp_file)
                exp_file.close()
                print('Saved sessions loaded from: sessions.pkl')
            except IOError:
               pass

        self.import_data()

        self.subject_total_rewards = self._get_total_rewards()

        if rebuild_sessions:
            self.save()

    def save(self):
        'Save sessions from experiment.'
        exp_file = open(os.path.join(self.path, 'sessions.pkl'),'wb')
        pickle.dump(self.sessions, exp_file)
        exp_file.close()

    def save_item(self, item, file_name):
        'save an item to experiment folder using pickle.'
        f = open(self.path + file_name + '.pkl', 'wb')
        pickle.dump(item, f)
        f.close()

    def load_item(self, item_name):
        'Unpickle and return specified item from experiment folder.'
        f = open(self.path + item_name + '.pkl', 'rb')
        out = pickle.load(f)
        f.close()
        return out

    def import_data(self):
        '''Load new sessions as session class instances.'''

        old_files = [session.file_name for session in self.sessions]
        files = os.listdir(self.path)
        new_files = [f for f in files if f not in old_files
                     and f[0].isdigit() and f[-4:] == '_log']

        if len(new_files) > 0:
            print('Loading new data files...')
            new_sessions = [hs.human_session(file_name,self.path) 
                            for file_name in new_files]

            self.sessions = self.sessions + new_sessions  

        self.n_subjects = len(np.unique([session.subject_ID for session in self.sessions]))
        self.subject_IDs= list(set([s.subject_ID for s in self.sessions]))
        self.n_days = max([session.number for session in self.sessions]) 

    def get_sessions(self, sIDs, numbers = []):
        '''Return list of sessions which match specified animal IDs and session numbers.
        All days or animals can be selected with input 'all'.
        The last n days can be selected with days = -n .
        '''
        if isinstance(sIDs, int): sIDs = [sIDs]
        if isinstance(numbers, int): numbers = [numbers]
        if numbers == 'all':
            numbers = list(set([s.number for s in self.sessions]))
        if sIDs == 'all':
            sIDs = self.subject_IDs
        valid_sessions = [s for s in self.sessions if s.number in numbers and s.subject_ID in sIDs]
        return valid_sessions    

    def _get_total_rewards(self):
        sub_tot_reward = {}
        for sID in self.subject_IDs:
            sub_tot_reward[sID] = sum([s.rewards for s in self.get_sessions(sID,'all')])
        return sub_tot_reward            

    # Plotting.
    def plot_subject(self, sID): pl.plot_subject(self, sID)