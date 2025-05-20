import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()

print(f"CUDA version used by PyTorch: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

print("\nTesting torch.fft.fft on CUDA...")
try:
    # Тест с комплексными числами
    a_complex = torch.randn(4, 4, dtype=torch.complex64, device='cuda')
    b_complex = torch.fft.fft(a_complex)
    print("torch.fft.fft (complex input) on CUDA test successful.")

    # Тест с вещественными числами (rfft используется для DCT)
    a_real = torch.randn(4, 4, device='cuda')
    b_real = torch.fft.rfft(a_real) # rfft для реальных входов
    print("torch.fft.rfft (real input) on CUDA test successful.")

except RuntimeError as e:
    print(f"CUDA FFT test failed: {e}")
    if "CUFFT_INTERNAL_ERROR" in str(e):
        print("CUFFT_INTERNAL_ERROR confirmed in basic test.")
except Exception as e:
    print(f"An unexpected error occurred during FFT test: {e}")