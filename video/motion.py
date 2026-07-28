from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

@dataclass
class MotionFrame:
    time: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    left_arm: float
    right_arm: float
    torso_sway: float

class GestureGenerator:
    """Deterministic beat-gesture motion generator, suitable as a motion-control track or training target."""
    def generate(self, text: str, duration: float, emotion: str="neutral", fps: int=30) -> list[MotionFrame]:
        if duration <= 0 or fps <= 0: raise ValueError("duration and fps must be positive")
        energy={"joy":1.3,"anger":1.2,"surprise":1.1,"sadness":.45,"fear":.65,"calm":.35,"neutral":.75}.get(emotion,.75)
        emphasis=max(1,len([w for w in text.split() if len(w)>5])); beat=max(.35,duration/(emphasis+2)); frames=[]
        for i in range(round(duration*fps)):
            t=i/fps; phase=2*math.pi*t/beat; envelope=.35+.65*(.5+.5*math.sin(phase))
            frames.append(MotionFrame(t,energy*6*math.sin(phase*.43),energy*4*math.sin(phase*.31),energy*2*math.sin(phase*.23),energy*34*envelope*math.sin(phase),-energy*34*envelope*math.sin(phase+.4),energy*3*math.sin(phase*.17)))
        return frames
    @staticmethod
    def to_numpy(frames: list[MotionFrame]) -> np.ndarray:
        return np.array([[f.time,f.head_yaw,f.head_pitch,f.head_roll,f.left_arm,f.right_arm,f.torso_sway] for f in frames],dtype=np.float32)
