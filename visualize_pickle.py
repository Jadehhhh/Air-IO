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

with open("AirIMU_EuRoC/net_output.pickle", "rb") as f:
    data = pickle.load(f)

inspect(data)
seq = data["MH_02_easy"]

print("ts shape:", seq["ts"].shape)
print("net_vel shape:", seq["net_vel"].shape)
print("cov shape:", seq["cov"].shape)

print("\nfirst 10 ts:")
print(seq["ts"][:10])

print("\nfirst 10 net_vel:")
print(seq["net_vel"][:10])

print("\nfirst 10 cov:")
print(seq["cov"][:10])