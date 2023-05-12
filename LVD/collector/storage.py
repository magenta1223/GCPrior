from collections import deque, OrderedDict
from copy import deepcopy

from ..contrib.simpl.collector.storage import  Batch, Buffer
import numpy as np
import torch
from torch.nn import functional as F




class Batch_Hstep(Batch):
    """
    TODO
    Batch에 H-step 이후의 transition을 가져오는 method를 만들고
    state는 맨 뒤의 H-step을 잘라버리면 그만임. 

    2) init methods
        - super().init(*args, **kwargs)
        - self.data (OrderedDict) 에
            - next_H_states : states를 H-step만큼 앞에서 잘라낸 데이터를 할당
            -('next_H_states', next_H_states) 

    3) property로 next_H_states 를 선언
        @property
        def next_H_states(self):
            return self.data['next_H_states']

    """

    def __init__(self, states, next_H_states, relabeled_goal, transitions=None):
        # super().__init__(states, actions, rewards, dones, next_states, transitions)
        self.data = OrderedDict([
            ('states', states),
            ('next_H_states', next_H_states),
            ('relabeled_goal', relabeled_goal)
        ])
        self.transitions = transitions
        self.H = 10
        # self.data['next_H_states'] = next_H_states # instance of collections.OrderedDict 
        # self.data.move_to_end('next_H_states', last = True) # reorder

    @property
    def next_H_states(self):
        return self.data['next_H_states']

    @property
    def relabeled_goal(self):
        return self.data['relabeled_goal']


class Buffer_H(Buffer):
    """
    Override 
    H-step state를 다.. 얻어놔야 함. 
    enqueue해서 다 얻어놨고, 이게.. H-step을 쓸 수 있는게 있고 아닌게 있음. 
    그냥 따로 구성하는게 .. 
    """
    def __init__(self, state_dim, action_dim, max_size, tanh = False, skimo = False):

        # if tanh:
        #     # normal action, action, loc, scale  
        #     super().__init__(state_dim, action_dim * 4, max_size)
        # else:
        #     super().__init__(state_dim, action_dim, max_size)


        if not skimo and tanh:
            super().__init__(state_dim, action_dim * 4, max_size)
            print(state_dim, action_dim)
        elif not skimo and not tanh:
            super().__init__(state_dim, action_dim * 3, max_size)
        else:
            super().__init__(state_dim, action_dim, max_size)

        self.H = 10
        self.H_ptr = 0
        self.H_size = 0

        self.H_episodes = deque()
        self.H_episode_ptrs = deque()
        self.H_transitions = torch.empty(max_size, 3*state_dim)
    
        dims = OrderedDict([
            ('state', state_dim),
            ('next_H_state', state_dim),
            ('relabeled_goal', state_dim),
        ])
        self.H_layout = dict()
        prev_i = 0
        for k, v in dims.items():
            next_i = prev_i + v
            self.H_layout[k] = slice(prev_i, next_i)
            prev_i = next_i

                

    @property
    def next_H_states(self):
        return self.H_transitions[:, self.H_layout['next_H_state']]
    
    @property
    def relabeled_goal(self):
        return self.H_transitions[:, self.H_layout['relabeled_goal']]

    # override 
    def enqueue(self, episode):
        super().enqueue(episode)
        """
        # 지금 내가 원하는 것
        현 시점 + 목표 => H-step 이후를 보고싶다.
        지금 episode가 high-episode란말야?
        그럼 transition이 (s_t, a_t, s_t+1)이 아니고
        (s_t, z, s_t+H) 임
        즉, next_states가 바로 H states
        여기서 H번 더한거는 H^2 이후임. 당연히 당연히 당연히 ~ 안된다. 
        state reconstruction이나, subgoal generation은 raw episode 상에서 수행해야 한다.
        """

        # raw_episode = episode.raw_episode

        # # 모든 raw_episode의 goal을 체크 
        # states = deepcopy(raw_episode.states)
        
        # # achieved goal을 return함
        # # 이게 변하는 마지막 순간이 last rwd index 
        # achieved = 0
        # goal_index = 0
        # for i, state in enumerate(states):
        #     achieved_now = len(GOAL_CHECKERS[env_name](state))
        #     if achieved_now > achieved:
        #         achieved = achieved_now
        #         goal_index = i

        # if goal_index != 0:
        #     self.enqueue_H(raw_episode, goal_index)
        
        
        
    def enqueue_H(self, raw_episode, goal_index):
        while len(self.H_episodes) > 0:
            old_episode = self.H_episodes[0]
            ptr = self.H_episode_ptrs[0]
            dist = (ptr - self.H_ptr) % self.max_size

            if dist < len(raw_episode):
                self.H_episodes.popleft()
                self.H_episode_ptrs.popleft()
            else:
                break
        self.H_episodes.append(raw_episode)
        self.H_episode_ptrs.append(self.H_ptr)
        


        states = deepcopy(raw_episode.states)[ : goal_index - self.H] # 
        relabeled_goal = raw_episode.states[goal_index]
        relabeled_goal[:9] = 0    
        states = np.array(states)

        # transition으로 만듬
        transitions = torch.as_tensor(np.concatenate([
            states[:-self.H], # states
            states[self.H:], # next H states
            relabeled_goal[np.newaxis, :].repeat(len(states) - self.H, axis = 0)              # Relabeled Goal임. 
        ], axis=-1))

        
        if len(transitions):
            if self.H_ptr + len(transitions) <= self.max_size:
                self.H_transitions[self.H_ptr:self.H_ptr+len(transitions)] = transitions
            elif self.H_ptr + len(transitions) < 2*self.max_size:
                self.H_transitions[self.H_ptr:] = transitions[:self.max_size-self.H_ptr]
                self.H_transitions[:len(transitions)-self.max_size+self.H_ptr] = transitions[self.max_size-self.H_ptr:]
            else:
                raise NotImplementedError

            # 즉, ptr은 현재 episode를 더하고 난 후의 위치임. 
            self.H_ptr = (self.H_ptr + len(transitions)) % self.max_size
            self.H_size = min(self.H_size + len(transitions), self.max_size)



    def sample(self, n):
        indices = torch.randint(self.size, size=[n], device=self.device)
        transitions = self.transitions[indices]
        return Batch(*[transitions[:, i] for i in self.layout.values()], transitions)

    def sample_Hstep(self, n):
        """
        별로 필요 없을 것 같은데 ? 
        for finetune subgoal generator 
        """

        if self.H_size > 0:
            # indices = torch.randint(self.size - self.H, size=[n], device=self.device)
            indices = torch.randint(self.H_size, size=[n], device=self.device)
            transitions = self.H_transitions[indices]
            return Batch_Hstep(*[transitions[:, i] for i in self.H_layout.values()], transitions)

        else:
            return None


class Offline_Buffer:
    def __init__(self, state_dim, action_dim, trajectory_length = 10, max_size = 1000) -> None:

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trajectory_length = trajectory_length
        self.max_size = max_size

        self.size = 0 
        self.pos = 0  

        self.states = torch.empty(max_size, trajectory_length + 1, state_dim)
        self.actions = torch.empty(max_size, trajectory_length, action_dim)

    def enqueue(self, states, actions):
        N, T, _ = actions.shape
        self.size = min(  self.size + N, self.max_size)        
        # if exceed max size
        if self.max_size < self.pos + N:
            self.states[self.pos : self.max_size] = states[: self.max_size - self.pos]
            self.actions[self.pos : self.max_size] = actions[: self.max_size - self.pos]
            self.pos = 0
            # remainder 
            states = states[self.max_size - self.pos : ]
            actions = actions[self.max_size - self.pos : ]

        N = states.shape[0]
        self.states[self.pos : self.pos + N] = states
        self.actions[self.pos : self.pos + N] = actions

        self.pos += N


    def sample(self):
        i = np.random.randint(0, self.size)

        states = self.states[i].numpy()
        actions = self.actions[i].numpy()

        return states, actions


    def copy_from(self, buffer):
        self.states = buffer.states.clone()
        self.actions = buffer.actions.clone()
        self.size = buffer.size
        self.pos = buffer.pos
        print(f"Buffer Size : {self.size}")
        
    def reset(self):
        
        self.size = 0 # 현재 buffer에 차 있는 subtrajectories의 전체 길이
        self.pos = 0  # 다음에 어디에 추가할지. 

        self.states = torch.empty(self.max_size, self.trajectory_length + 1, self.state_dim)
        self.actions = torch.empty(self.max_size, self.trajectory_length, self.action_dim)
        print(  "Buffer Reset. Size : ", self.size)