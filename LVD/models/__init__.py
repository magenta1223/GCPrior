
from .sc import StateConditioned_Model
from .gc import GoalConditioned_Model
from .sc_div import StateConditioned_Diversity_Model
from .gc_div import GoalConditioned_Diversity_Model
from .gc_div_joint import GoalConditioned_Diversity_Joint_Model
from .gc_div_joint_gp import GoalConditioned_GoalPrompt_Model
from .skimo import Skimo_Model
from .WAE import WAE
from .simpl import SiMPL_Model


MODELS = {
    "sc" : StateConditioned_Model,
    "sc_dreamer" : StateConditioned_Model,
    "simpl" : SiMPL_Model,
    "gc" : GoalConditioned_Model,
    "sc_div" : StateConditioned_Diversity_Model,
    "gc_div" : GoalConditioned_Diversity_Model,
    "gc_div_joint" : GoalConditioned_Diversity_Joint_Model,
    "gc_div_joint_gp" : GoalConditioned_GoalPrompt_Model,
    "skimo" : Skimo_Model,
    "gc_skimo" : Skimo_Model,
    "WAE": WAE,
}