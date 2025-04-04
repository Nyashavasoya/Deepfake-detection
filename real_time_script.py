import numpy as np
import scipy.io.wavfile
from models.AASIST import Model
import subprocess
import wave
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def load_aasist_model(checkpoint_path):
    print(f"[INFO] Loading model from {checkpoint_path}")
    
    model_config = {
        "nb_samp": 64000,  # Updated to match our target length
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
    
    print(f"[DEBUG] Model configuration: {model_config}")
    
    try:
        model = Model(model_config)
        print(f"[INFO] Model instance created")
        
        # Print model structure
        print(f"[DEBUG] Model architecture:\n{model}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Total model parameters: {total_params:,}")
        
        # Load weights
        print(f"[INFO] Loading weights from checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] Weights loaded successfully")
        
        model.to(device)
        print(f"[INFO] Model moved to device: {device}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

model = load_aasist_model("aasist_trained.pth")

def preprocess_audio(file_path, target_sample_rate=16000, target_length=64000):
    try:
        waveform, sr = torchaudio.load(file_path)
        print(f"[DEBUG] Loaded file {file_path} - Shape: {waveform.shape}, SR: {sr}")

        # Resample if needed
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
            print(f"[DEBUG] Resampled to {target_sample_rate}Hz")

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"[DEBUG] Converted to mono - New shape: {waveform.shape}")

        # Ensure correct length (4 seconds)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]  # Trim
        else:
            pad_size = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))  # Pad

        print(f"[DEBUG] Final waveform shape: {waveform.shape}")
        return waveform.squeeze()
    
    except Exception as e:
        print(f"[ERROR] Failed to process audio {file_path}: {e}")
        return None


def infer(file_path):
    print(f"[INFO] Running inference on {file_path}")
    waveform = preprocess_audio(file_path)
    
    if waveform is None:
        return "Error in processing audio"

    waveform = waveform.to(device).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        _, output = model(waveform)  # Forward pass
        prediction = torch.argmax(output, dim=1).item()

    label = "spoof" if prediction == 1 else "bona-fide"
    print(f"[RESULT] Predicted label: {label}")
    return label

# Example usage
# audio_file = "path/to/audio.wav"
# result = infer(audio_file)
# print(f"Inference Result: {result}")
def record_audio(duration=4, sample_rate=16000, output_file="live_audio.wav"):
    print("[INFO] Recording...")

    # Use ffmpeg to record audio (Make sure it's installed)
    cmd = [
        "ffmpeg", "-f", "alsa", "-t", str(duration),
        "-ac", "1", "-ar", str(sample_rate),
        "-i", "default", output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[INFO] Recording saved as {output_file}")
    return output_file


# Example: Record & classify
live_audio_path = record_audio()
live_result = infer(live_audio_path)  # Assuming 'infer' is your deepfake detection function
print(f"Real-Time Inference Result: {live_result}")

