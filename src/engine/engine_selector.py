import platform

def get_engine():
    system = platform.system()
    processor = platform.processor()
    
    # Check if we are on a Mac with Apple Silicon
    is_mac_arm = system == 'Darwin' and 'arm' in processor.lower()
    
    if is_mac_arm:
        print("** Mac Detected: Loading CoreML Engine...")
        from src.engine.coreml_engine import CoreMLEngine
        return CoreMLEngine()
    else:
        print("** Cloud Detected: Loading PyTorch Engine...")
        from src.engine.pytorch_engine import PyTorchEngine
        return PyTorchEngine()