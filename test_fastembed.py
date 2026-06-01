from fastembed import TextEmbedding
import onnxruntime as ort

print("ONNX Runtime available providers:", ort.get_available_providers())

try:
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", providers=["CUDAExecutionProvider"])
    print("Successfully instantiated fastembed with CUDA.")
except Exception as e:
    print(f"Failed to instantiate fastembed with CUDA: {e}")
