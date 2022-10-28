import importlib

def get_default(problem_name, sb_param):
    env_name = problem_name.lower()
    config_fn = env_name + "_" + sb_param.replace("-", "_")

    # print(env_name)
    module = importlib.import_module(f"configs.{env_name}")
    assert hasattr(module, config_fn)
    return getattr(module, config_fn)()
