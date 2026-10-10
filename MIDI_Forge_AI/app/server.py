"""
MIDI Forge AI - Local Desktop Server
Self-contained Python HTTP server listening on loopback (127.0.0.1).
Serves the native HTML/CSS/JS frontend and provides full REST endpoints
for MIDI analysis, dataset preparation, training, generation, and diagnostics.
"""

import os
import sys
import json
import time
import urllib.parse
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.midi.parser import MidiParser, NoteEvent, MidiFileInfo
from app.midi.writer import MidiWriter
from app.midi.synthetic_dataset import generate_synthetic_dataset
from app.engine.hybrid_adapter import HybridMusicAdapter
from app.training.model import MusicGPT, GPTConfig
from app.training.dataset import DatasetBuilder, MusicDataset
from app.training.trainer import LocalMusicTrainer
from app.generation.generator import MidiGenerator
from app.storage.project_manager import ProjectManager

try:
    import torch
except ImportError:
    torch = None


class AppState:
    """Singleton holding shared application state across requests."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.project_mgr = ProjectManager(root_dir)
        self.parser = MidiParser()
        self.writer = MidiWriter(ticks_per_beat=480)
        self.generator = MidiGenerator(output_dir=self.project_mgr.output_dir)
        self.trainer = LocalMusicTrainer(checkpoints_dir=self.project_mgr.checkpoints_dir)

        self.current_adapter: Optional[HybridMusicAdapter] = None
        self.current_dataset: Optional[MusicDataset] = None
        self.current_model: Optional[MusicGPT] = None
        self.active_model_name: str = "default_model"

        # Check if synthetic dataset exists; if not, generate on first start
        dataset_files = [f for f in os.listdir(self.project_mgr.datasets_dir) if f.endswith(".mid")]
        if not dataset_files:
            generate_synthetic_dataset(self.project_mgr.datasets_dir)


def make_handler_class(root_dir: str):
    state = AppState(root_dir)
    frontend_dir = os.path.join(root_dir, "frontend")

    class MidiForgeHandler(BaseHTTPRequestHandler):
        def _send_json(self, data: Any, status: int = 200):
            response_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(response_bytes)

        def _send_error(self, message: str, status: int = 400):
            self._send_json({"error": message, "status": "error"}, status=status)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            # API Endpoints
            if path == "/api/status":
                device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                gpu_name = torch.cuda.get_device_name(0) if (torch and torch.cuda.is_available()) else None
                self._send_json({
                    "app": "MIDI Forge AI",
                    "version": "1.0.0",
                    "torch_available": torch is not None,
                    "torch_version": torch.__version__ if torch else None,
                    "device": device,
                    "gpu_name": gpu_name,
                    "training_status": state.trainer.get_progress()["status"],
                })
                return

            if path == "/api/stats":
                self._send_json(state.project_mgr.get_stats())
                return

            if path == "/api/library":
                files = state.project_mgr.list_library_files()
                self._send_json({"files": files, "count": len(files)})
                return

            if path == "/api/training/progress":
                self._send_json(state.trainer.get_progress())
                return

            if path == "/api/output":
                output_files = []
                for f in sorted(os.listdir(state.project_mgr.output_dir), reverse=True):
                    if f.lower().endswith(".mid"):
                        fp = os.path.join(state.project_mgr.output_dir, f)
                        info, _ = state.parser.parse_file(fp)
                        d = info.to_dict()
                        output_files.append(d)
                self._send_json({"files": output_files, "count": len(output_files)})
                return

            if path == "/api/history":
                self._send_json(state.project_mgr.get_generation_history())
                return

            if path == "/api/diagnostics/run":
                from tests.test_suite import run_all_tests
                test_summary = run_all_tests()
                self._send_json(test_summary)
                return

            if path == "/api/download_portable_zip":
                from package_distribution import create_portable_zip, OUTPUT_ZIP
                if not os.path.exists(OUTPUT_ZIP):
                    create_portable_zip()
                with open(OUTPUT_ZIP, "rb") as f:
                    zip_data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="MIDI_Forge_AI_Windows_Portable.zip"')
                self.send_header("Content-Length", str(len(zip_data)))
                self.end_headers()
                self.wfile.write(zip_data)
                return

            if path == "/api/download":
                filename = query.get("file", [""])[0]
                if not filename or ".." in filename:
                    self._send_error("Nome de arquivo inválido", 400)
                    return
                # Check output dir or dataset dir
                target_path = os.path.join(state.project_mgr.output_dir, filename)
                if not os.path.exists(target_path):
                    target_path = os.path.join(state.project_mgr.datasets_dir, filename)
                if not os.path.exists(target_path):
                    self._send_error("Arquivo não encontrado", 404)
                    return

                with open(target_path, "rb") as f:
                    file_data = f.read()

                self.send_response(200)
                self.send_header("Content-Type", "audio/midi")
                self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(target_path)}"')
                self.send_header("Content-Length", str(len(file_data)))
                self.end_headers()
                self.wfile.write(file_data)
                return

            # Serve static frontend files
            rel_file = path.lstrip("/")
            if not rel_file or rel_file == "/":
                rel_file = "index.html"

            file_path = os.path.join(frontend_dir, rel_file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                mime, _ = mimetypes.guess_type(file_path)
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime or "text/plain")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                # Fallback to index.html for SPA routes
                index_path = os.path.join(frontend_dir, "index.html")
                if os.path.exists(index_path):
                    with open(index_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self._send_error("Arquivo não encontrado", 404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}

            if path == "/api/library/synthetic":
                files = generate_synthetic_dataset(state.project_mgr.datasets_dir)
                self._send_json({"status": "success", "created_files": [os.path.basename(f) for f in files]})
                return

            if path == "/api/library/upload":
                # Expects { "filename": str, "content_base64": str }
                import base64
                fname = data.get("filename", f"upload_{int(time.time())}.mid")
                if not fname.endswith((".mid", ".midi")):
                    fname += ".mid"
                b64 = data.get("content_base64", "")
                if not b64:
                    self._send_error("Conteúdo vazio")
                    return
                raw_bytes = base64.b64decode(b64)
                target = os.path.join(state.project_mgr.datasets_dir, os.path.basename(fname))
                with open(target, "wb") as f:
                    f.write(raw_bytes)
                info, _ = state.parser.parse_file(target)
                self._send_json({"status": "success", "file": info.to_dict()})
                return

            if path == "/api/datasets/prepare":
                mode = data.get("mode", "remi")
                block_size = int(data.get("block_size", 256))
                augment = bool(data.get("augment_transposition", False))

                adapter = HybridMusicAdapter(mode=mode)
                state.current_adapter = adapter

                # Collect all valid midis from datasets dir
                midi_files = []
                for root, _, files in os.walk(state.project_mgr.datasets_dir):
                    for f in files:
                        if f.lower().endswith((".mid", ".midi")):
                            midi_files.append(os.path.join(root, f))

                builder = DatasetBuilder(parser=state.parser)
                prep_result = builder.prepare_dataset(
                    midi_filepaths=midi_files,
                    adapter=adapter,
                    block_size=block_size,
                    augment_transposition=augment
                )
                state.current_dataset = prep_result["dataset"]
                self._send_json({"status": "success", "report": prep_result["report"]})
                return

            if path == "/api/models/create":
                model_name = data.get("name", f"model_{int(time.time())}")
                vocab_size = int(data.get("vocab_size", state.current_adapter.vocab_size if state.current_adapter else 512))
                block_size = int(data.get("block_size", 256))
                n_layer = int(data.get("n_layer", 4))
                n_head = int(data.get("n_head", 4))
                n_embd = int(data.get("n_embd", 128))

                config = GPTConfig(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    dropout=float(data.get("dropout", 0.1))
                )
                model = MusicGPT(config)
                state.current_model = model
                state.active_model_name = model_name

                # Save model config metadata
                meta_path = os.path.join(state.project_mgr.models_dir, f"{model_name}.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "name": model_name,
                        "created_at": time.time(),
                        "config": config.to_dict(),
                        "num_params": model.get_num_params(),
                    }, f, indent=2)

                self._send_json({
                    "status": "success",
                    "name": model_name,
                    "num_params": model.get_num_params(),
                    "config": config.to_dict()
                })
                return

            if path == "/api/training/start":
                epochs = int(data.get("epochs", 8))
                batch_size = int(data.get("batch_size", 4))
                lr = float(data.get("learning_rate", 0.0005))
                model_name = data.get("model_name", state.active_model_name)

                # Ensure adapter & dataset are ready
                if state.current_dataset is None:
                    # Auto-prepare if not done yet
                    adapter = HybridMusicAdapter(mode="remi")
                    state.current_adapter = adapter
                    midi_files = [
                        os.path.join(state.project_mgr.datasets_dir, f)
                        for f in os.listdir(state.project_mgr.datasets_dir)
                        if f.endswith(".mid")
                    ]
                    builder = DatasetBuilder(parser=state.parser)
                    res = builder.prepare_dataset(midi_files, adapter=adapter, block_size=128)
                    state.current_dataset = res["dataset"]

                # Ensure model is ready
                if state.current_model is None:
                    cfg = GPTConfig(
                        vocab_size=state.current_adapter.vocab_size,
                        block_size=128,
                        n_layer=3,
                        n_head=4,
                        n_embd=128,
                    )
                    state.current_model = MusicGPT(cfg)

                try:
                    state.trainer.start_training(
                        model=state.current_model,
                        dataset=state.current_dataset,
                        model_name=model_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=lr,
                    )
                    self._send_json({"status": "success", "message": "Treinamento iniciado."})
                except Exception as e:
                    self._send_error(str(e))
                return

            if path == "/api/training/pause":
                state.trainer.pause()
                self._send_json({"status": "success", "message": "Treinamento pausado."})
                return

            if path == "/api/training/resume":
                state.trainer.resume()
                self._send_json({"status": "success", "message": "Treinamento retomado."})
                return

            if path == "/api/training/stop":
                state.trainer.stop()
                self._send_json({"status": "success", "message": "Interrupção solicitada."})
                return

            if path == "/api/generation/generate":
                temperature = float(data.get("temperature", 0.95))
                top_k = int(data.get("top_k", 40))
                top_p = float(data.get("top_p", 0.9))
                max_tokens = int(data.get("max_tokens", 200))
                seed = int(data["seed"]) if "seed" in data and data["seed"] is not None else None
                ref_file = data.get("reference_file", None)
                ref_path = os.path.join(state.project_mgr.datasets_dir, os.path.basename(ref_file)) if ref_file else None

                # Ensure model & adapter exist
                if state.current_adapter is None:
                    state.current_adapter = HybridMusicAdapter(mode="remi")
                if state.current_model is None:
                    cfg = GPTConfig(
                        vocab_size=state.current_adapter.vocab_size,
                        block_size=256,
                        n_layer=3,
                        n_head=4,
                        n_embd=128,
                    )
                    state.current_model = MusicGPT(cfg)

                gen_result = state.generator.generate(
                    model=state.current_model,
                    adapter=state.current_adapter,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    reference_midi_path=ref_path,
                    context_retention_pct=float(data.get("context_retention_pct", 0.3)),
                    bpm=float(data.get("bpm", 120.0)),
                    device="cpu"
                )

                state.project_mgr.add_generation_record(gen_result)
                self._send_json(gen_result)
                return

            if path == "/api/diagnostics/run":
                from tests.test_suite import run_all_tests
                test_summary = run_all_tests()
                self._send_json(test_summary)
                return

            self._send_error("Endpoint não encontrado", 404)

        def log_message(self, format, *args):
            # Suppress noisy GET log output for smooth background running
            pass

    return MidiForgeHandler


def run_server(host: str = "127.0.0.1", port: int = 8080, root_dir: Optional[str] = None):
    root_dir = root_dir or PROJECT_ROOT
    handler_class = make_handler_class(root_dir)
    HTTPServer.allow_reuse_address = True
    server = HTTPServer((host, port), handler_class)
    print(f"============================================================")
    print(f"  MIDI FORGE AI - SERVIDOR LOCAL EM EXECUÇÃO")
    print(f"  Acesse: http://{host}:{port}")
    print(f"  Diretório raiz: {root_dir}")
    print(f"  Pressione Ctrl+C para encerrar com segurança.")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor MIDI Forge AI com segurança...")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MIDI Forge AI Local Desktop Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--root", default=PROJECT_ROOT, help="Root directory")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, root_dir=args.root)
