import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

def test_imports():
    """Test importing essential third-party libraries."""
    print("[1/4] Checking python library imports...", flush=True)
    
    modules_to_test = [
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("sklearn", "scikit-learn"),
        ("streamlit", "Streamlit"),
    ]
    
    missing = []
    for mod_name, label in modules_to_test:
        try:
            __import__(mod_name)
            print(f"  ✓ {label} ({mod_name}) successfully imported.", flush=True)
        except ImportError:
            print(f"  ✗ {label} ({mod_name}) NOT found.", flush=True)
            missing.append(label)
            
    return missing

def test_pytorch_device():
    """Test PyTorch availability and hardware acceleration support."""
    print("\n[2/4] Checking PyTorch setup and computing device...", flush=True)
    try:
        import torch
        print(f"  - PyTorch Version: {torch.__version__}", flush=True)
        
        if torch.cuda.is_available():
            device = "cuda"
            print("  - Hardware Acceleration: CUDA GPU detected.", flush=True)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("  - Hardware Acceleration: Apple Silicon (MPS) detected.", flush=True)
        else:
            device = "cpu"
            print("  - Hardware Acceleration: CPU mode.", flush=True)
            
        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"  ✓ Tensor allocated on device: '{device}'. Output: {tensor.tolist()}", flush=True)
    except Exception as e:
        print(f"  ✗ PyTorch device test failed: {e}", flush=True)

def test_project_configuration():
    """Test configuration settings loading."""
    print("\n[3/4] Checking project settings and path resolution...", flush=True)
    try:
        from config import settings
        print(f"  - Base Directory: {settings.BASE_DIR}", flush=True)
        print(f"  - Data Directory: {settings.DATA_DIR}", flush=True)
        print(f"  - Models Directory: {settings.MODELS_DIR}", flush=True)
        print(f"  - App Environment: {settings.APP_ENV}", flush=True)
        print("  ✓ Configuration settings loaded cleanly.", flush=True)
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}", flush=True)

def main():
    print("========================================", flush=True)
    print(" AI Crop Doctor - Setup Verification ", flush=True)
    print("========================================\n", flush=True)
    
    missing_packages = test_imports()
    test_pytorch_device()
    test_project_configuration()
    
    print("\n[4/4] Final Verdict:", flush=True)
    if missing_packages:
        print(f"  ⚠ Missing packages detected: {', '.join(missing_packages)}", flush=True)
        print("  Run 'pip install -r requirements.txt' to install missing dependencies.", flush=True)
    else:
        print("  🎉 Setup test passed! All core components and paths are ready.", flush=True)
    print("========================================", flush=True)

if __name__ == "__main__":
    main()
