import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import json
import pyloudnorm as pyln
import warnings
from openai import OpenAI
from dotenv import load_dotenv

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Load API Key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_PATH = "yamnet_genre_model.keras"
MAPPING_PATH = "yamnet_mapping.json"
TARGET_SR = 16000 

# STATISTICAL TARGETS
GENRE_TARGETS = {
    "TRAP":  {"low": 0.45, "mid_low": 0.20, "mid_high": 0.15, "high": 0.20, "lufs": -9.0, "dynamic_range": 5.0},
    "DRILL": {"low": 0.50, "mid_low": 0.15, "mid_high": 0.15, "high": 0.20, "lufs": -8.5, "dynamic_range": 6.0},
    "POP":   {"low": 0.25, "mid_low": 0.25, "mid_high": 0.30, "high": 0.20, "lufs": -9.0, "dynamic_range": 8.0},
    "EDM":   {"low": 0.35, "mid_low": 0.20, "mid_high": 0.25, "high": 0.20, "lufs": -6.0, "dynamic_range": 4.5},
    "ROCK":  {"low": 0.20, "mid_low": 0.35, "mid_high": 0.30, "high": 0.15, "lufs": -10.0, "dynamic_range": 7.0},
    "RnB":   {"low": 0.30, "mid_low": 0.30, "mid_high": 0.25, "high": 0.15, "lufs": -10.0, "dynamic_range": 8.0},
    "default": {"low": 0.30, "mid_low": 0.25, "mid_high": 0.25, "high": 0.20, "lufs": -10.0, "dynamic_range": 8.0}
}

class AudioAnalyzer:
    def __init__(self):
        print("Loading YAMNet & Classifier...")
        try:
            self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
            self.model = tf.keras.models.load_model(MODEL_PATH)
            with open(MAPPING_PATH, "r") as f:
                self.mapping = json.load(f)["genres"]
            print("Analyzer Ready.")
        except Exception as e:
            print(f"Error loading models: {e}", flush=True)
            self.model = None

    # MATH LAYER (Feature Extraction)
    def check_stereo_image(self, file_path):
        try:
            duration = librosa.get_duration(filename=file_path)
            start = max(0, duration/2 - 5)
            y_stereo, _ = librosa.load(file_path, sr=44100, mono=False, offset=start, duration=10)
            if y_stereo.ndim == 1: return 0.0, "Mono"
            L, R = y_stereo[0], y_stereo[1]
            mid, side = (L + R) / 2, (L - R) / 2
            width = np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-9)
            return float(width), "Stereo"
        except: return 0.5, "Unknown"

    def check_dynamics(self, y):
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        peak = np.max(np.abs(y))
        return float(20 * np.log10(peak / (np.mean(rms) + 1e-9)))

    def analyze_spectrum(self, y, sr):
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        total = np.sum(S) + 1e-9
        return {
            "low":      float(np.sum(S[freqs < 200]) / total),
            "mid_low":  float(np.sum(S[(freqs >= 200) & (freqs < 2000)]) / total),
            "mid_high": float(np.sum(S[(freqs >= 2000) & (freqs < 8000)]) / total),
            "high":     float(np.sum(S[freqs >= 8000]) / total)
        }

    # --- 2. GPT CONSULTANT LAYER ---
    def consult_gpt(self, raw_findings, genre):
        """
        Sends the mathematical proof to GPT and asks for a humane explanation in JSON.
        """
        system_prompt = """
        You are a world-class Mixing Engineer mentor. 
        You will receive a list of mathematical mixing errors detected in a user's track.
        Your job is to explain WHY these are issues for the specific genre and suggest a fix.
        
        OUTPUT FORMAT: Return ONLY valid JSON. Structure:
        {
            "summary": "A 1-sentence friendly summary of the mix status.",
            "issues": {
                "mix_balance": [{"issue": "Title", "reason": "Explanation", "fix": "Action"}],
                "dynamics": [],
                "loudness": [],
                "stereo": []
            }
        }
        Keep the tone encouraging but professional. Be specific about Hz and dB.
        """
        
        user_prompt = f"""
        Genre: {genre}
        
        DETECTED MATH DEVIATIONS:
        {json.dumps(raw_findings, indent=2)}
        
        If the list is empty, return an empty JSON structure with a 'Great Job' summary.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # mini for speed (1-2s latency)
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"GPT Error: {e}")
            # Fallback if GPT fails
            return {
                "summary": "AI Analysis complete (Offline Mode).",
                "issues": {"mix_balance": [], "dynamics": [], "loudness": [], "stereo": []}
            }

    def analyze(self, file_path):
        if not self.model: return {"error": "Model not loaded"}

        try:
            # A. AI CLASSIFICATION
            wav, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            _, embeddings, _ = self.yamnet(wav)
            frame_predictions = self.model.predict(embeddings, verbose=0)
            avg_prediction = np.mean(frame_predictions, axis=0)
            predicted_index = np.argmax(avg_prediction)
            predicted_genre = self.mapping[predicted_index]
            confidence = round(float(avg_prediction[predicted_index]) * 100, 1)

            # B. FEATURE EXTRACTION
            y_high, sr_high = librosa.load(file_path, sr=44100)
            user_stats = {
                "spectrum": self.analyze_spectrum(y_high, sr_high),
                "dynamic_range": self.check_dynamics(y_high),
                "stereo_width": self.check_stereo_image(file_path)[0],
                "lufs": float(pyln.Meter(sr_high).integrated_loudness(y_high))
            }
            target = GENRE_TARGETS.get(predicted_genre, GENRE_TARGETS["default"])

            # C. DETECT MATH DEVIATIONS (The "Proof")
            raw_findings = {"mix_balance": [], "dynamics": [], "loudness": [], "stereo": []}
            
            # Spectrum Check
            for band, name in [("low", "Sub-Bass"), ("mid_low", "Low-Mids"), ("high", "Highs")]:
                diff = user_stats["spectrum"][band] - target[band]
                if abs(diff) > 0.08:
                    raw_findings["mix_balance"].append(f"{name} is {int(diff*100)}% {'louder' if diff>0 else 'quieter'} than target.")

            # Dynamics Check
            dyn_diff = user_stats["dynamic_range"] - target["dynamic_range"]
            if dyn_diff < -2.5: raw_findings["dynamics"].append(f"Dynamic Range is {user_stats['dynamic_range']:.1f}dB (Target {target['dynamic_range']}dB). Too Compressed.")
            elif dyn_diff > 4.0: raw_findings["dynamics"].append("Too loose/uncompressed.")

            # Loudness Check
            lufs_diff = user_stats["lufs"] - target["lufs"]
            if abs(lufs_diff) > 3.0: 
                raw_findings["loudness"].append(f"Loudness is {user_stats['lufs']:.1f} LUFS (Target {target['lufs']} LUFS).")

            # Stereo Check
            if user_stats["stereo_width"] < 0.15: 
                raw_findings["stereo"].append(f"Stereo Width is {user_stats['stereo_width']:.2f} (Near Mono).")

            # D. PASS TO GPT
            print("sending data to GPT...")
            gpt_feedback = self.consult_gpt(raw_findings, predicted_genre)

            # E. RETURN MERGED DATA
            return {
                "meta": {
                    "genre": str(predicted_genre),
                    "confidence": float(confidence),
                    "filename": os.path.basename(file_path),
                    "advice_summary": gpt_feedback["summary"]
                },
                "metrics": {
                    "lufs": round(user_stats["lufs"], 1),
                    "stereo_width": round(user_stats["stereo_width"], 2),
                    "dynamic_range": round(user_stats["dynamic_range"], 1),
                    "mud_ratio": 0.0 # Placeholder
                },
                "visualization": {
                    "user_spectrum": user_stats["spectrum"],
                    "ideal_spectrum": {k:v for k,v in target.items() if k in user_stats["spectrum"]},
                    "confidence_curve": [float(x) for x in np.max(frame_predictions, axis=1).tolist()][:50]
                },
                "issues": gpt_feedback["issues"] # This is now GPT-generated text!
            }

        except Exception as e:
            print(f"ANALYZER ERROR: {str(e)}")
            return {"error": str(e)}

    def check_mud(self, y, sr):
        # ... keep helper if needed ...
        return 0.0

if __name__ == "__main__":
    pass