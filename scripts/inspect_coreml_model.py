import inspect
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

print(" Inspecting CoreMLModel source:")
try:
    print(inspect.getsource(CoreMLModel))
except Exception as e:
    print(f"Could not get source: {e}")
    # Fallback: dir()
    print("Dir(CoreMLModel):", dir(CoreMLModel))
