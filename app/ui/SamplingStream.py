import numpy as np
import threading
import queue
import time
import wave

# ============================================================================================================#
#                                                Sampling Motor                                               #
#                                                                                                             #
#  - Drives a background reader thread that streams fixed-size PCM chunks from a WAV file.                    #                                                               
#  - Keeps approximate real-time pacing based on sample rate and chunk length.                                # 
#  - Exposes start/pause/stop controls that operate on the reader thread and appstate flags.                  #  
# ============================================================================================================#
class SamplingStream:
    
    # Constructor
    def __init__(self, appstate):
        self.appstate = appstate
    
    # This method starts the background reader thread
    def start_stream(self):
        if not self.appstate.audio_file_path:
            return
        if self.appstate.reader_thread is None or not self.appstate.reader_thread.is_alive():
            self.appstate.stop_reader = False
            self.appstate.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
            self.appstate.reader_thread.start()
        self.appstate.paused = False
    
    # This method pauses the reader thread
    def pause_stream(self):
        self.appstate.paused = True

    # This method stops the reader thread
    def stop_stream(self):
        self.appstate.stop_reader = True
        if self.appstate.reader_thread and self.appstate.reader_thread.is_alive():
            self.appstate.reader_thread.join(timeout=0.5)
        self.appstate.reader_thread = None
        self.appstate.paused = True
        
    # This method runs in the background thread and reads audio data from the WAV file
    def reader_loop(self):
        try:
            with wave.open(self.appstate.audio_file_path, 'rb') as wf:
                
                # ---------- Read header metadata ----------
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()

                if sr != self.appstate.sample_rate:
                    self.appstate.sample_rate = sr
                    self.appstate.needs_reset = True  
                try:
                    wf.setpos(max(0, int(self.appstate.pos_frames)))
                except Exception:
                    self.appstate.pos_frames = 0

                frames_per_chunk = int(self.appstate.block_size)

                # ====================== Streaming loop ======================
                while not self.appstate.stop_reader:
                    if self.appstate.paused:
                        time.sleep(0.05)
                        continue
                    
                    # ---------- Read next chunk ----------
                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        self.appstate.paused = True
                        break
                    
                    if sw == 2:
                        chunk = np.frombuffer(data, dtype=np.int16)
                        if ch > 1:
                            chunk = chunk.reshape(-1, ch)[:, 0]
                        chunk = chunk.astype(np.float32) / 32768.0
                    elif sw == 1:
                        chunk = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                        if ch > 1:
                            chunk = chunk.reshape(-1, ch)[:, 0]
                        chunk = (chunk - 128.0) / 128.0
                    elif sw == 4:
                        chunk = np.frombuffer(data, dtype=np.float32)
                        if ch > 1:
                            chunk = chunk.reshape(-1, ch)[:, 0]
                    else:
                        print("[audio] formato WAV no soportado (samplewidth)", sw)
                        self.appstate.paused = True
                        break

                    try:
                        self.appstate.q.put_nowait(chunk)
                    except queue.Full:
                        try: _ = self.appstate.q.get_nowait()
                        except queue.Empty: pass
                        try: self.appstate.q.put_nowait(chunk)
                        except queue.Full: pass

                    time.sleep(len(chunk) / float(self.appstate.sample_rate))
                    self.appstate.pos_frames += len(chunk)
        finally:
            pass

