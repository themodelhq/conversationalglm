from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from torch.optim import AdamW
from audio.emotion import EMOTIONS, SpeechEmotionRecognizer
from audio.features import load_audio, log_mel
from vision.encoder import VisionEncoder

class ASRModel(nn.Module):
    def __init__(self, vocab: int = 256):
        super().__init__()
        self.frontend = nn.Sequential(nn.Conv1d(80, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, 256, 5, padding=2), nn.GELU())
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True), 4)
        self.head = nn.Linear(256, vocab)
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.frontend(mel.transpose(1, 2)).transpose(1, 2)))

class AcousticTTS(nn.Module):
    def __init__(self, vocab: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 256, padding_idx=0)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True), 4)
        self.mel = nn.Linear(256, 80)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mel(self.encoder(self.embedding(tokens)))

class MotionModel(nn.Module):
    def __init__(self, output_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, output_dim))
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

class LipSyncModel(nn.Module):
    def __init__(self, landmarks: int = 20):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(80, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, landmarks, 1))
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel.transpose(1, 2)).transpose(1, 2)

class VideoDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(3, 64, 3, padding=1), nn.GELU(), nn.Conv3d(64, 64, 3, padding=1), nn.GELU(), nn.Conv3d(64, 3, 3, padding=1))
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.net(video)

class VisionPretrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionEncoder(layers=6, heads=12)
        self.head = nn.Linear(768, 3)
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(pixels)[:, 0])

def resize_sequence(sequence: torch.Tensor, length: int) -> torch.Tensor:
    if sequence.shape[0] == length:
        return sequence
    return nn.functional.interpolate(sequence.T.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0).T

class ModalDataset(Dataset):
    def __init__(self, path: str | Path, task: str):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.task = task
        if not self.rows:
            raise ValueError("Training dataset is empty")
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        if self.task == "asr":
            waveform, _ = load_audio(row["audio_path"])
            return log_mel(waveform), torch.tensor(list(row["transcript"].encode("utf-8")), dtype=torch.long)
        if self.task == "tts":
            tokens = torch.tensor(list(row["transcript"].encode("utf-8")), dtype=torch.long)
            if not len(tokens):
                tokens = torch.tensor([1], dtype=torch.long)
            waveform, _ = load_audio(row["audio_path"])
            return tokens, resize_sequence(log_mel(waveform), len(tokens))
        if self.task == "lipsync":
            waveform, _ = load_audio(row["audio_path"])
            mel = log_mel(waveform)
            landmarks = torch.tensor(row["mouth_landmarks"], dtype=torch.float32)
            return mel, resize_sequence(landmarks, len(mel))
        if self.task == "emotion_recognition":
            waveform, _ = load_audio(row["audio_path"])
            emotion = str(row.get("emotion", "neutral"))
            return log_mel(waveform), torch.tensor(EMOTIONS.index(emotion) if emotion in EMOTIONS else 0)
        if self.task == "vision":
            from PIL import Image
            pixels = VisionEncoder.preprocess(Image.open(row["image_path"]))
            return pixels, pixels.mean(dim=(1, 2))
        if self.task == "video":
            capture = cv2.VideoCapture(row["video_path"]); frames = []
            while len(frames) < 16:
                valid, frame = capture.read()
                if not valid: break
                frame = cv2.resize(frame, (128, 128))[:, :, ::-1].copy()
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1)
            capture.release()
            if len(frames) < 2: raise ValueError(f"Video has fewer than two decodable frames: {row['video_path']}")
            return torch.stack(frames).permute(1, 0, 2, 3), torch.tensor(0)
        features = torch.tensor(row["features"], dtype=torch.float32)
        if features.shape != (256,): raise ValueError("motion, gesture, emotion_generation, and memory data require a 256-value features vector")
        dimensions = {"motion": 7, "gesture": 7, "emotion_generation": 8, "memory": 256}[self.task]
        target = torch.tensor(row["target"], dtype=torch.float32)
        if target.shape != (dimensions,): raise ValueError(f"{self.task} target requires {dimensions} values")
        return features, target

def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    def combine(values: tuple[torch.Tensor, ...], value: float = 0.0) -> torch.Tensor:
        if values[0].ndim in (1, 2): return nn.utils.rnn.pad_sequence(list(values), batch_first=True, padding_value=value)
        return torch.stack(list(values))
    return combine(xs), combine(ys, -100.0)

def build(task: str) -> nn.Module:
    return {"asr": ASRModel(), "tts": AcousticTTS(), "lipsync": LipSyncModel(), "emotion_recognition": SpeechEmotionRecognizer(), "vision": VisionPretrainer(), "video": VideoDenoiser(), "motion": MotionModel(7), "gesture": MotionModel(7), "emotion_generation": MotionModel(8), "memory": MotionModel(256)}[task]

def run(task: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True); parser.add_argument("--validation"); parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "no")
    model = build(task); loader = DataLoader(ModalDataset(args.train, task), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=0.01); model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train(); steps = 0; final_loss = float("nan")
    for _ in range(args.epochs):
        for x, y in loader:
            with accelerator.accumulate(model):
                output = model(x)
                if task == "asr":
                    targets = y.long(); target_lengths = targets.ne(-100).sum(-1); loss = nn.functional.ctc_loss(output.log_softmax(-1).transpose(0, 1), targets.masked_fill(targets.lt(0), 0), torch.full_like(target_lengths, output.shape[1]), target_lengths, blank=0, zero_infinity=True)
                elif task == "emotion_recognition": loss = nn.functional.cross_entropy(output, y.long())
                elif task == "video":
                    noise = torch.randn_like(x); loss = nn.functional.mse_loss(output(x + noise * 0.1), noise)
                else:
                    if output.ndim == 3: loss = nn.functional.l1_loss(output, y[:, :output.shape[1]])
                    else: loss = nn.functional.mse_loss(output, y)
                accelerator.backward(loss); accelerator.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(); steps += 1; final_loss = float(loss.detach().float())
    accelerator.wait_for_everyone(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), output / f"{task}.pt")
        (output / "training_summary.json").write_text(json.dumps({"task": task, "steps": steps, "final_loss": final_loss, "epochs": args.epochs}, indent=2))
if __name__ == "__main__": run("asr")
