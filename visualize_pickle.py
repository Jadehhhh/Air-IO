import pickle

def inspect(obj, name="root", depth=0, max_depth=2):
    indent = "  " * depth
    print(f"{indent}{name}: {type(obj)}")
    
    if depth >= max_depth:
        return
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{indent}  key = {k}")
            inspect(v, name=f"[{k}]", depth=depth+1, max_depth=max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}  length = {len(obj)}")
        if len(obj) > 0:
            inspect(obj[0], name="[0]", depth=depth+1, max_depth=max_depth)
    elif hasattr(obj, "shape"):
        print(f"{indent}  shape = {obj.shape}")

with open("experiments/euroc/motion_body_rot/net_output.pickle", "rb") as f:
    data = pickle.load(f)

inspect(data)