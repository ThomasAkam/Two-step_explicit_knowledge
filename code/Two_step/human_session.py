import os
import numpy as np
from collections import namedtuple
from . import plotting as pl

Event = namedtuple('Event', ['time','name'])

# Event IDs ------------------------------------------------

# Dict indicating how different events are named in data files.

event_IDs =     {'trial_start'  : 'ons_ms1',
                 'second_step'  : 'ons_ms2',
                 'reward'       : 'ons_msReward',
                 'non-reward'   : 'ons_msNoRwrd',
                 'choice-down'  : 'triggered Down',
                 'choice-up'    : 'triggered Up',
                 'choice-left'  : 'triggered Left',
                 'choice-right' : 'triggered Right', 
                 'press-down'   : 'down',
                 'press-up'     : 'up',
                 'press-left'   : 'left',
                 'press-right'  : 'right'}

ID2event = {v: k for k, v in event_IDs.items()}

class human_session():
    'Class containing data from a human session.'
    def __init__(self, file_name, data_path):
        # Session information.
        print(file_name)
        self.file_name  = file_name
        self.subject_ID =  int(file_name.split('_')[0])
        self.number = int(file_name.split('_')[1][7:])

        # Import data.
        with open(os.path.join(data_path, file_name), 'r') as data_file:
            data_lines = [line.strip() for line in data_file if line[0].isdigit()]

        #  Extract time stamps and data strings.

        time_stamps = np.array([float(line.split(' :')[0]) for line in data_lines])
        line_strings   = [line.split(' :')[1] for line in data_lines]

        trial_block_info = [ls for ls in line_strings if ls[:6] == 'trial-'] # Lines  carrying trial block info.

        # Store data and summary info.

        self.rewards  = sum([ls == event_IDs['reward'] for ls in line_strings])
        self.n_trials = len(trial_block_info)

        self.fraction_rewarded = self.rewards /  self.n_trials

        # -------------------------------------------------------------------------------------------
        # Make dictionary of choices, transitions, second steps and outcomes on each trial.
        #--------------------------------------------------------------------------------------------
 
        choice_strings      = [ls for ls in line_strings if ls in (event_IDs['choice-down'], event_IDs['choice-up'   ])]
        second_step_strings = [ls for ls in line_strings if ls in (event_IDs['choice-left'], event_IDs['choice-right'])]
        outcome_strings     = [ls for ls in line_strings if ls in (event_IDs['reward']     , event_IDs['non-reward'])]

        assert len(choice_strings) == len(second_step_strings) == len(outcome_strings) == self.n_trials, \
               'Unable to read file ' + file_name + ' as number of choices, second steps or outcomes does not match number of trials.'

        choices      = np.array([ c == event_IDs['choice-up']   for c in choice_strings], bool)      # True if high, flase if low.
        second_steps = np.array([ c == event_IDs['choice-left'] for c in second_step_strings], bool) # True if left, false if right.     
        outcomes     = np.array([ c == event_IDs['reward']   for c in outcome_strings], bool)        # True if rewarded,  flase if no reward.
        transitions  = choices == second_steps  # True if high --> left or low --> right,
                                                # flase if high --> right or low --> left.

        self.trial_data = {'choices'      : choices.astype(int), #Store as integer arrays.
                           'transitions'  : transitions.astype(int),
                           'second_steps' : second_steps.astype(int),
                           'outcomes'     : outcomes.astype(int)}

        # -------------------------------------------------------------------------------------------
        # Make list of events in the order they occured.
        #--------------------------------------------------------------------------------------------

        self.events = [Event(ts, ID2event[ls]) for (ts, ls) in zip(time_stamps, line_strings)
                       if ls in ID2event.keys()]

        # -------------------------------------------------------------------------------------------
        # Extract times of trial events.
        #--------------------------------------------------------------------------------------------
        
        def get_times(event_names):
            return np.array([ev.time for ev in self.events if ev.name in event_names])

        self.times = {event_name: get_times(event_name) for event_name in event_IDs.keys()}

        self.times.update({'choice'      : get_times(['choice-down', 'choice-up']),
                           'ss_action'   : get_times(['choice-left','choice-right'])})

        #--------------------------------------------------------------------------------------------
        # Extract block information.
        #--------------------------------------------------------------------------------------------

        trial_rew_state     = np.array([int(t.split('/')[1].split(':')[1]) for t in trial_block_info], int) # Reward state for each trial.
        trial_trans_state = np.array([int(t.split('/')[2].split(':')[1]) for t in trial_block_info], bool)  # Transition state for each trial.
        block_start_mask = (np.not_equal(trial_rew_state[1:], trial_rew_state[:-1]) | np.not_equal(trial_trans_state[1:], trial_trans_state[:-1]))
        start_trials = [0] + (np.where(block_start_mask)[0] + 1).astype(int).tolist() # Block start trials (trials numbered from 0).
        transition_states = trial_trans_state[np.array(start_trials)].astype(int)     # Transition state for each block.
        reward_states = trial_rew_state[np.array(start_trials)]                       # Reward state for each block.
                
        self.blocks = {'start_trials'      : start_trials,
                       'end_trials'        : start_trials[1:] + [self.n_trials],
                       'start_times'       : None,
                       'reward_states'     : reward_states,      # 0 for left good, 1 for neutral, 2 for right good.
                       'transition_states' : transition_states,  # 1 for A blocks, 0 for B blocks.
                       'trial_trans_state' : trial_trans_state,
                       'trial_rew_state'   : trial_rew_state}  


    def select_trials(self, selection_type, select_n=20, block_type='all'):
        ''' Select specific trials for analysis.  
        'end' : Only final select_n trials of each block are selected.
        'xtr' : Select all trials except select_n trials following transition reversal.
        'all' : All trials are included.
        'rng' : A range of trials relative to block start.
        The block_type argument allows additional selection for only 'neutral' or 'non_neutral' blocks.
        '''
        assert selection_type in ['end', 'xtr', 'all', 'rng'], 'Invalid trial select type.'

        if selection_type == 'xtr': # Select all trials except select_n following transition reversal.
            trials_to_use = np.ones(self.n_trials, dtype = bool)
            trans_change = np.hstack((
                False, ~np.equal(self.blocks['transition_states'][:-1],
                                 self.blocks['transition_states'][1:])))
            start_trials = (self.blocks['start_trials'] + 
                            [self.blocks['end_trials'][-1] + select_n])
            for b in range(len(trans_change)):
                if trans_change[b]:
                    trials_to_use[start_trials[b]:start_trials[b] + select_n] = False

        elif selection_type == 'end': # Select only select_n trials before block transitions.
            trials_to_use = np.zeros(self.n_trials, dtype = bool)
            for t in self.blocks['start_trials'][1:]:
                trials_to_use[t-1-select_n:t-1] = True

        elif selection_type == 'all': # Use all trials.
            trials_to_use = np.ones(self.n_trials, dtype = bool)

        elif selection_type == 'rng': # Select a range of trials relative to block start.
          trials_to_use = np.zeros(self.n_trials, dtype = bool)
          for t in self.blocks['start_trials']:
            trials_to_use[t+select_n[0]:t+select_n[1]] = True

        if not block_type == 'all': #  Restrict analysed trials to blocks of certain types.
            if block_type == 'neutral':       # Include trials only from neutral blocks.
                block_selection = self.blocks['trial_rew_state'] == 1
            elif block_type == 'non_neutral': # Include trials only from non-neutral blocks.
                block_selection = self.blocks['trial_rew_state'] != 1
            trials_to_use = trials_to_use & block_selection

        return trials_to_use

    #------------------------------------------------------------------------------------------------

    def plot(self, fig_no = 1): pl.plot_session(self, fig_no)

    def unpack_trial_data(self, order = 'CTSO', dtype = int):
        'Return elements of trial_data dictionary in specified order and data type.'
        o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
        if dtype == int:
            return [self.trial_data[o_dict[i]] for i in order]
        else:
            return [self.trial_data[o_dict[i]].astype(dtype) for i in order]