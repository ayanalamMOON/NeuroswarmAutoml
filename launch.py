"""
NeuroSwarm-AutoML Interactive Execution Launcher.

Provides a CLI menu to launch AutoML search pipelines, start the web server,
run hardware latency benchmarks, export TensorRT engines, and compile reports.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
os.environ["PYTHONPATH"] = str(SRC_DIR) + os.path.pathsep + os.environ.get("PYTHONPATH", "")


def print_header():
    print("=" * 68)
    print("🛸  NEUROSWARM-AUTOML EXECUTION LAUNCHER")
    print("    Bi-Level Co-Evolutionary Search & Acceleration Suite")
    print("=" * 68)


def get_latest_output_dir() -> str:
    candidates = [p for p in ROOT_DIR.glob("outputs_*") if p.is_dir()]
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)
    return "./outputs_cifar100_cuda"


def run_automl_search():
    print("\n--- 🛸 Run AutoML Architecture Search ---")
    dataset = input("Select Dataset (cifar10 / cifar100 / synthetic) [default: cifar100]: ").strip() or "cifar100"
    generations = input("Number of Generations [default: 5]: ").strip() or "5"
    population = input("Population Size [default: 8]: ").strip() or "8"
    use_ray = input("Enable Ray Distributed Execution? (y/n) [default: y]: ").strip().lower() != "n"

    cmd = [
        sys.executable,
        str(SRC_DIR / "neuroswarm" / "main.py"),
        "--dataset", dataset,
        "--generations", generations,
        "--population", population,
        "--output_dir", f"./outputs_{dataset}_run"
    ]
    if use_ray:
        cmd.append("--use_ray")

    print(f"\n[EXEC] Executing: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def launch_web_server():
    print("\n--- 🚀 Launch Web Server & Gradio UI ---")
    default_dir = get_latest_output_dir()
    onnx_files = list(Path(default_dir).glob("*.onnx")) if Path(default_dir).exists() else []
    default_model = str(onnx_files[0]) if onnx_files else "./outputs_cifar100_cuda/winner_4f5974bd.onnx"

    model_path = input(f"Model path (.onnx or .pt) [default: {default_model}]: ").strip() or default_model
    port = input("Server Port [default: 8000]: ").strip() or "8000"

    cmd = [
        sys.executable,
        str(SRC_DIR / "neuroswarm" / "utils" / "server.py"),
        "--model", model_path,
        "--device", "cuda",
        "--port", port
    ]
    print(f"\n[EXEC] Server starting at http://localhost:{port}/ui ...\n")
    subprocess.run(cmd)


def run_benchmarks():
    print("\n--- ⚡ Hardware Latency Benchmarks ---")
    print("1. Benchmark PyTorch CUDA Model (.pt)")
    print("2. Benchmark Native TensorRT Engine (.engine)")
    choice = input("Select Benchmark Option [1-2]: ").strip()

    if choice == "1":
        pt_model = input("Path to .pt model [default: ./outputs_cifar100_cuda/winner_4f5974bd.pt]: ").strip() or "./outputs_cifar100_cuda/winner_4f5974bd.pt"
        cmd = [sys.executable, str(SRC_DIR / "neuroswarm" / "utils" / "benchmark_pt.py"), "--model", pt_model]
    else:
        trt_engine = input("Path to .engine model [default: ./outputs_trt/winner_4f5974bd.engine]: ").strip() or "./outputs_trt/winner_4f5974bd.engine"
        cmd = [sys.executable, str(SRC_DIR / "neuroswarm" / "utils" / "benchmark_trt.py"), "--engine", trt_engine]

    subprocess.run(cmd)


def export_tensorrt():
    print("\n--- 🛠️ Export ONNX Model to TensorRT Engine ---")
    onnx_path = input("Path to input .onnx model [default: ./outputs_cifar100_cuda/winner_4f5974bd.onnx]: ").strip() or "./outputs_cifar100_cuda/winner_4f5974bd.onnx"

    cmd = [
        sys.executable,
        str(SRC_DIR / "neuroswarm" / "utils" / "trt_exporter.py"),
        "--onnx", onnx_path,
        "--fp16",
        "--output_dir", "./outputs_trt"
    ]
    subprocess.run(cmd)


def generate_report():
    print("\n--- 📊 Generate Experiment Executive Report ---")
    target_dir = input("Output directory to summarize [default: ./outputs_cifar100_cuda]: ").strip() or "./outputs_cifar100_cuda"

    cmd = [
        sys.executable,
        str(SRC_DIR / "neuroswarm" / "utils" / "reporter.py"),
        "--output_dir", target_dir
    ]
    subprocess.run(cmd)


def main():
    while True:
        print_header()
        print("1. 🛸 Run AutoML Search Engine")
        print("2. 🚀 Launch FastAPI Server & Gradio UI")
        print("3. ⚡ Run CUDA / TensorRT Latency Benchmarks")
        print("4. 🛠️ Export Model to TensorRT FP16 Engine")
        print("5. 📊 Compile Experiment REPORT.md Summary")
        print("6. ❌ Exit")
        print("-" * 68)

        choice = input("Enter option [1-6]: ").strip()

        if choice == "1":
            run_automl_search()
        elif choice == "2":
            launch_web_server()
        elif choice == "3":
            run_benchmarks()
        elif choice == "4":
            export_tensorrt()
        elif choice == "5":
            generate_report()
        elif choice == "6":
            print("\nExiting NeuroSwarm-AutoML. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1-6.")

        input("\nPress Enter to return to the main menu...")
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
