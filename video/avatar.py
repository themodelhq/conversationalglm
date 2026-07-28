from __future__ import annotations
from pathlib import Path
import json
from video.generation import VideoGenerator
from video.lipsync import LipSyncEngine
from video.motion import GestureGenerator

class AvatarPipeline:
    def __init__(self, video_generator: VideoGenerator | None=None, lipsync: LipSyncEngine | None=None, gestures: GestureGenerator | None=None):
        self.video_generator=video_generator or VideoGenerator(); self.lipsync=lipsync or LipSyncEngine(); self.gestures=gestures or GestureGenerator()
    def render(self, visual_prompt: str, speech_audio: str | Path, spoken_text: str, output: str | Path, emotion: str="neutral", duration: float=4.0) -> Path:
        output=Path(output); base=output.with_name(output.stem+"_base.mp4"); self.video_generator.generate(f"{visual_prompt}, {emotion} facial expression, natural hand gestures",base,frames=max(8,round(duration*8)),fps=8)
        motion=self.gestures.generate(spoken_text,duration,emotion); output.with_suffix('.motion.json').write_text(json.dumps([f.__dict__ for f in motion]))
        return self.lipsync.sync(base,speech_audio,output)
