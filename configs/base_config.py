from easydict import EasyDict as edict


class BaseConfig:
    """
    dummy class for configuration
    """
    
    def __init__(self):
        pass

    def set_attrs(self, dictionary):
        if isinstance(dictionary, dict):
            dictionary = edict(dictionary)
        self.attrs = dictionary

    def get(self, name):
        if not name : #name is None or not name:
            return None
        return getattr(self.attrs, name)

    def log(self):
        pass

    def load(self, path):
        pass

    def __repr__(self):
        print(f"{self.name.upper()} specification")
        print("#" + "-" * 20 + "#" )
        for k, v in self.attrs.items():
            print(f"{k} : {v}")
        return self.name
        
    def __str__(self):
        return self.name

    def get_build(self, config):
        return NotImplementedError

class SequentialModelConfig(BaseConfig):
    """
    dummy class for configuration
    """
    
    def __init__(self):
        super().__init__()


    def layerbuild(self, attr_list, repeat = None):
        build =  [[ self.get(attr_nm) for attr_nm in attr_list  ]]
        if repeat is not None:
            build = build * repeat
        return build 




class BaseDataConfig(BaseConfig):
    def __init__(self):
        super().__init__()