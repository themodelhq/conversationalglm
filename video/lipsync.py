from __future__ import annotations
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import cv2
import librosa

class LipSyncEngine:
    """Audio-driven mouth-opening compositor. For photorealistic use, swap the exported track into a Wav2Lip/SadTalker renderer."""
    def mouth_track(self, audio_path: str | Path, fps: int=30) -> np.ndarray:
        audio,sr=librosa.load(str(audio_path),sr=16000,mono=True)
        hop=max(1,round(sr/fps)); rms=librosa.feature.rms(y=audio,frame_length=hop*2,hop_length=hop)[0]
        rms=(rms-rms.min())/(np.ptp(rms)+1e-8); return np.clip(rms**.65,0,1)
    def sync(self, video_path: str | Path, audio_path: str | Path, output_path: str | Path, fps: int=30) -> Path:
        video_path, audio_path, output_path=map(Path,(video_path,audio_path,output_path)); output_path.parent.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(video_path)); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); source_fps=cap.get(cv2.CAP_PROP_FPS) or fps
        track=self.mouth_track(audio_path,round(source_fps)); temp=Path(tempfile.mkstemp(suffix='.mp4')[1]); writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),source_fps,(width,height)); cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        i=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            faces=cascade.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.15,5)
            if len(faces):
                x,y,w,h=max(faces,key=lambda a:a[2]*a[3]); openness=float(track[min(i,len(track)-1)]) if len(track) else 0.
                cx,cy=x+w//2,y+int(h*.72); mw,mh=int(w*.18),max(2,int(h*.018+openness*h*.09))
                overlay=frame.copy(); cv2.ellipse(overlay,(cx,cy),(mw,mh),0,0,360,(25,10,35),-1); frame=cv2.addWeighted(overlay,.42,frame,.58,0)
            writer.write(frame); i+=1
        cap.release(); writer.release()
        subprocess.run(["ffmpeg","-y","-i",str(temp),"-i",str(audio_path),"-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",str(output_path)],check=True,capture_output=True); temp.unlink(missing_ok=True); return output_path
