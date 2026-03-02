from deeplearning_library import Parameter

class Module: 
    def __init__(self):
        self.training = True
        self.layer = NotImplementedError

    def train(self):
        self.training = True
        for m in self.submodules(): m.train()

    def eval(self):
        self.training = False
        for m in self.submodules(): m.eval()

    def __call__(self, input):
        return self.forward(input)

    def forward(self,x):
        return self.layer.forward(x)
    
    def backward(self,loss):
        loss.backward(self.parameters())

    def parameters(self) -> list:
        params = []
        seen = set()

        def get_params(obj):
            if isinstance(obj, Parameter):
                if obj not in seen:
                    params.append(obj)
                    seen.add(obj)
            elif isinstance(obj, Module):
                for attr in obj.__dict__.values():
                    get_params(attr)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    get_params(item)

        get_params(self)
        return params
    
    def state_dict(self, prefix=""):
        sd = {}
        for name, attr in self.__dict__.items():
            # Cas d'un paramètre (feuille)
            if isinstance(attr, Parameter):
                sd[prefix + name] = attr.tensor.to_numpy()
            
            # Cas d'un sous-module (branche)
            elif isinstance(attr, Module):
                sd.update(attr.state_dict(prefix + name + "."))
                
            # Cas de ta liste dans Sequential
            elif isinstance(attr, list) and name == "layers":
                for i, layer in enumerate(attr):
                    if isinstance(layer, Module):
                        sd.update(layer.state_dict(prefix + f"{i}."))
        return sd
    
    def load_state_dict(self, state_dict: dict, prefix=""):
        """
        Charge récursivement les poids dans les paramètres et sous-modules.
        """
        for name, attr in self.__dict__.items():
            # 1. Cas d'un paramètre direct (ex: self.weight)
            if isinstance(attr, Parameter):
                key = prefix + name
                if key in state_dict:
                    # On utilise ta méthode Rust .set()
                    attr.set(state_dict[key])
                else:
                    print(f"Warning: Key '{key}' not found in state_dict.")

            # 2. Cas d'un sous-module (ex: un bloc custom)
            elif isinstance(attr, Module):
                attr.load_state_dict(state_dict, prefix=prefix + name + ".")

            # 3. Cas particulier de ta liste 'layers' dans Sequential
            elif isinstance(attr, list) and name == "layers":
                for i, layer in enumerate(attr):
                    if isinstance(layer, Module):
                        layer.load_state_dict(state_dict, prefix=prefix + f"{i}.")