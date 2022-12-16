from proposed.contrib.simpl.collector.storage import  Batch, Buffer
import torch

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

    def __init__(self, states, actions, rewards, dones, next_states, next_H_states, transitions=None):
        super().__init__(states, actions, rewards, dones, next_states, transitions)
        self.H = 10
        self.data['next_H_states'] = next_H_states # instance of collections.OrderedDict 
        self.data.move_to_end('next_H_states', last = True) # reorder

    @property
    def next_H_states(self):
         return self.data['next_H_states']


class Buffer(Buffer):
    """
    Override 
    H-step state를 다.. 얻어놔야 함. 
    enqueue해서 다 얻어놨고, 이게.. H-step을 쓸 수 있는게 있고 아닌게 있음. 
    그냥 따로 구성하는게 .. 
    """
    def __init__(self, state_dim, action_dim, max_size, batch_cls = Buffer):
        super().__init__(state_dim, action_dim, max_size)
        self.batch_cls = batch_cls
        self.H = 10
        # 여기서 Hstep 정보를 미리 만들어줘야 함. 
        
    
    def sample(self, n):
        indices = torch.randint(self.size, size=[n], device=self.device)
        transitions = self.transitions[indices]
        return Batch(*[transitions[:, i] for i in self.layout.values()], transitions)

    def sample_Hstep(self, n):
        """
        for latent state consistency
        last H step은 H step 이후의 데이터가 없음. 
        이걸 제외하고 해야됨. 따라서 randint의 sampling 범위에 제한을 건다.
        """

        indices = torch.randint(self.size - self.H, size=[n], device=self.device)
        transitions = self.transitions[indices]
        H_transitions = self.transitions[indices + self.H]
        return Batch_Hstep(*[transitions[:, i] for i in self.layout.values()], H_transitions[:, self.layout['state']], transitions)



