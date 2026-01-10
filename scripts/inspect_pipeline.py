import inspect
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline

print(" Inspecting CoreMLStableDiffusionPipeline.__call__ source:")
try:
    print(inspect.getsource(CoreMLStableDiffusionPipeline.__call__))
except Exception as e:
    print(e)
