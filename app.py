"""
🎤 Audio Deepfake Analyzer - Web Version (Veritas.AI)
======================================================
Flask backend with all features from GUI v3:
- Record audio or upload files
- CNN (ResNet) based deepfake detection
- Comprehensive explainability with legends
- Mixed voice detection
- Noise analysis
- Research references

Author: Thesis Project
Date: December 2025
"""

import os
import sys
import json
import base64
import io
import uuid
import gc
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import librosa
import soundfile as sf
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import entropy
from PIL import Image

# Gemini AI Service for explanations
from geminiservice import get_gemini_service, GeminiService
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# Add parent directory to path for model access
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Some saved Keras models include Lambda layers that reference functions defined in
# the local training module (e.g., `cnn_resnet.py`). Importing it here ensures the
# module is available during deserialization.
try:
    import cnn_resnet  # noqa: F401
except Exception:
    cnn_resnet = None

# Serve React build from 'dist' folder
app = Flask(__name__, static_folder='dist', static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# ==========================================
# ⚙️ SETTINGS
# ==========================================
SAMPLE_RATE = 16000
DEFAULT_DURATION = 3
DURATION_OPTIONS = [3, 5, 7, 10, 15, 20, 30]
SNR_LEVELS = [5, 10, 15, 20, 25, 30]
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Colors for visualization (matching GUI)
COLORS = {
    'bg_card': '#1a1a2e',
    'accent': '#4361ee',
    'accent_light': '#4cc9f0',
    'success': '#06d6a0',
    'danger': '#ef476f',
    'warning': '#ffd166',
}

# Global model storage
loaded_model = None
model_type = "CNN"
model_path = None

# Backup inference ("ResNet v2" button in frontend)
HF_RESNET_V2_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
_hf_resnet_v2_pipeline = None

# Ollama LLM for plain-English explanations (LEGACY - kept for fallback)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# ==========================================
# 🤖 GEMINI AI CONFIGURATION
# ==========================================
# Set your Gemini API key here or via environment variable GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyCn68Hxi29jba4exHJXeCdAbcFMxqzWDx4')  # Uses env var or fallback

# Initialize Gemini service (will be None if no API key)
gemini_service: GeminiService = None


# ==========================================
# 📚 RESEARCH REFERENCES
# ==========================================
RESEARCH_REFERENCES = {
    'waveform': [
        {'name': 'Waveform Analysis in Audio Processing - IEEE', 'url': 'https://ieeexplore.ieee.org/document/6296526'},
        {'name': 'Audio Signal Processing Fundamentals', 'url': 'https://www.sciencedirect.com/topics/engineering/audio-signal-processing'}
    ],
    'spectrogram': [
        {'name': 'Mel Spectrogram for Deepfake Detection - ASVspoof', 'url': 'https://www.asvspoof.org/'},
        {'name': 'Audio Deepfake Detection Using Spectrograms - arXiv', 'url': 'https://arxiv.org/abs/2003.09725'},
        {'name': 'Librosa: Audio Analysis in Python', 'url': 'https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html'}
    ],
    'saliency': [
        {'name': 'Grad-CAM: Visual Explanations from Deep Networks', 'url': 'https://arxiv.org/abs/1610.02391'},
        {'name': 'Explainable AI for Audio Classification', 'url': 'https://arxiv.org/abs/2105.05658'},
        {'name': 'Saliency Maps for CNN Interpretation', 'url': 'https://arxiv.org/abs/1312.6034'}
    ],
    'mfcc': [
        {'name': 'MFCC in Speaker Recognition - IEEE', 'url': 'https://ieeexplore.ieee.org/document/1163420'},
        {'name': 'Audio Deepfake Detection with MFCC', 'url': 'https://arxiv.org/abs/2009.02446'},
        {'name': 'Librosa MFCC Documentation', 'url': 'https://librosa.org/doc/main/generated/librosa.feature.mfcc.html'}
    ],
    'confidence': [
        {'name': 'Confidence Calibration in Deep Learning', 'url': 'https://arxiv.org/abs/1706.04599'},
        {'name': 'Uncertainty Estimation in Neural Networks', 'url': 'https://arxiv.org/abs/1703.04977'}
    ],
    'voice_analysis': [
        {'name': 'Speaker Diarization - Google Research', 'url': 'https://arxiv.org/abs/1810.04719'},
        {'name': 'Multi-Speaker Detection - Interspeech', 'url': 'https://www.isca-speech.org/archive/interspeech_2020/'},
        {'name': 'Pitch Tracking with Librosa', 'url': 'https://librosa.org/doc/main/generated/librosa.piptrack.html'}
    ],
    'noise': [
        {'name': 'Noise Robustness in Audio Deepfake Detection', 'url': 'https://arxiv.org/abs/2107.14567'},
        {'name': 'ASVspoof 2021: Noise & Channel Effects', 'url': 'https://arxiv.org/abs/2109.00535'},
        {'name': 'SNR-Aware Training for Robust Detection', 'url': 'https://ieeexplore.ieee.org/document/9414231'}
    ],
    'resnet': [
        {'name': 'Deep Residual Learning (ResNet) - CVPR', 'url': 'https://arxiv.org/abs/1512.03385'},
        {'name': 'Transfer Learning for Audio', 'url': 'https://arxiv.org/abs/1912.10211'},
        {'name': 'CNN for Audio Deepfake Detection', 'url': 'https://arxiv.org/abs/2004.03831'}
    ],
    'real_verdict': [
        {'name': 'Characteristics of Genuine Speech', 'url': 'https://arxiv.org/abs/2004.05130'},
        {'name': 'Human vs Synthetic Voice Analysis', 'url': 'https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf'}
    ],
    'fake_verdict': [
        {'name': 'Detecting AI-Generated Speech - Google', 'url': 'https://arxiv.org/abs/2004.06053'},
        {'name': 'TTS Artifacts in Deepfakes', 'url': 'https://arxiv.org/abs/2003.09725'},
        {'name': 'ASVspoof Challenge Overview', 'url': 'https://www.asvspoof.org/'}
    ]
}


# ==========================================
# 🔧 HELPER FUNCTIONS
# ==========================================

def find_default_model():
    """Find default model path"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    default_paths = [
        os.path.join(base_dir, "resnet_fast_model.keras"),
        os.path.join(base_dir, "CNN_recomputed", "resnet_fast_model.keras"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path
    return None


def load_model_file(path, mtype="CNN"):
    """Load model from path"""
    global loaded_model, model_type, model_path
    
    try:
        if mtype == "CNN":
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Load with safe_mode=False to handle Lambda layer properly
            try:
                loaded_model = tf.keras.models.load_model(path, safe_mode=False, compile=False)
                print("   ✅ Model loaded with safe_mode=False")
            except TypeError:
                # Fallback for older Keras versions
                loaded_model = tf.keras.models.load_model(path, compile=False)
                print("   ✅ Model loaded with compile=False (legacy)")
            
            # Re-compile the model
            loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            loaded_model = joblib.load(path)
        
        model_type = mtype
        model_path = path
        return True, os.path.basename(path)
    except Exception as e:
        return False, str(e)


def compute_spectrogram(audio):
    """Compute mel spectrogram for CNN"""
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    
    stft = tf.signal.stft(audio_tensor, frame_length=320, frame_step=32, fft_length=512)
    spectrogram = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=224,
        num_spectrogram_bins=tf.shape(spectrogram)[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=8000.0
    )
    
    mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    mel_spec = tf.expand_dims(mel_spec, -1)
    mel_spec = tf.image.resize(mel_spec, (224, 224))
    
    mel_min = tf.reduce_min(mel_spec)
    mel_max = tf.reduce_max(mel_spec)
    mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + 1e-8)
    
    return mel_spec.numpy()


def compute_mfcc(audio):
    """Compute MFCC features"""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])


def analyze_voices(audio):
    """Analyze audio for multiple voices/sources with better accuracy"""
    analysis = {
        'num_sources': 1,
        'source_type': 'single_voice',
        'confidence': 0.0,
        'details': [],
        'warning': None,
        'pitch_data': [],  # For visualization
        'explanation': ''
    }
    
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        
        pitch_values = []
        pitch_times = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                pitch_times.append(t * 512 / SAMPLE_RATE)
        
        if len(pitch_values) > 0:
            pitch_values = np.array(pitch_values)
            pitch_times = np.array(pitch_times)
            
            # Store for visualization
            analysis['pitch_data'] = {
                'times': pitch_times.tolist(),
                'values': pitch_values.tolist()
            }
            
            pitch_range = np.max(pitch_values) - np.min(pitch_values)
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            
            # More sophisticated histogram analysis
            hist, bin_edges = np.histogram(pitch_values, bins=30)
            hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')
            peaks, peak_props = find_peaks(hist_smooth, height=np.max(hist_smooth) * 0.25, distance=3)
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
            spectral_std = np.std(spectral_centroids)
            spectral_mean = np.mean(spectral_centroids)
            
            rms = librosa.feature.rms(y=audio)[0]
            rms_peaks, _ = find_peaks(rms, distance=15, prominence=0.1*np.max(rms))
            
            # Calculate coefficient of variation for pitch (more robust than just range)
            pitch_cv = (pitch_std / pitch_mean) * 100 if pitch_mean > 0 else 0
            
            # Indicators with explanations
            multiple_voice_indicators = 0
            details = []
            explanations = []
            
            # Check for distinct pitch clusters (more strict)
            if len(peaks) >= 3:
                multiple_voice_indicators += 1.5
                details.append(f"Multiple distinct pitch clusters ({len(peaks)} groups)")
                explanations.append("Multiple separate pitch frequency groups suggest different speakers")
            elif len(peaks) == 2:
                # Check if peaks are far apart (different vocal ranges)
                peak_centers = [(bin_edges[p] + bin_edges[p+1])/2 for p in peaks]
                if len(peak_centers) >= 2 and abs(peak_centers[1] - peak_centers[0]) > 80:
                    multiple_voice_indicators += 1
                    details.append(f"Two distinct pitch ranges detected")
                    explanations.append("Two separate pitch ranges may indicate male/female voices or different speakers")
            
            # More conservative pitch range check (human voice typically 85-300Hz, but varies)
            # Single speaker can have 100-150Hz range naturally
            if pitch_range > 300:
                multiple_voice_indicators += 1
                details.append(f"Very wide pitch range ({pitch_range:.0f} Hz)")
                explanations.append("Pitch range exceeds typical single-speaker variation (usually <200Hz)")
            elif pitch_range > 200:
                # Only count if CV is also high
                if pitch_cv > 25:
                    multiple_voice_indicators += 0.5
                    details.append(f"Wide pitch range with high variation ({pitch_range:.0f} Hz, CV={pitch_cv:.1f}%)")
            
            # Spectral variation (more conservative threshold)
            if spectral_std > 800:
                multiple_voice_indicators += 1
                details.append(f"High spectral variation (σ={spectral_std:.0f})")
                explanations.append("Significant changes in voice timbre suggest different voice characteristics")
            
            # Energy peaks (more conservative)
            audio_duration = len(audio) / SAMPLE_RATE
            peaks_per_second = len(rms_peaks) / audio_duration
            if peaks_per_second > 5:
                multiple_voice_indicators += 0.5
                details.append(f"Frequent energy peaks ({len(rms_peaks)} peaks, {peaks_per_second:.1f}/sec)")
                explanations.append("Frequent energy variations may indicate speaker changes")
            
            # Make decision (more conservative threshold)
            if multiple_voice_indicators >= 2.5:
                analysis['num_sources'] = 2
                analysis['source_type'] = 'multiple_voices'
                analysis['confidence'] = min(0.95, 0.5 + multiple_voice_indicators * 0.12)
                analysis['details'] = details
                analysis['explanation'] = " | ".join(explanations) if explanations else "Multiple acoustic indicators suggest different voice sources"
                analysis['warning'] = "⚠️ MULTIPLE VOICES DETECTED: This audio may contain multiple speakers."
            else:
                analysis['num_sources'] = 1
                analysis['source_type'] = 'single_voice'
                analysis['confidence'] = max(0.7, 0.95 - multiple_voice_indicators * 0.1)
                
                # Provide explanation for single voice too
                single_details = []
                if pitch_range <= 150:
                    single_details.append(f"Consistent pitch range ({pitch_range:.0f} Hz)")
                if pitch_cv < 20:
                    single_details.append(f"Low pitch variation (CV={pitch_cv:.1f}%)")
                if len(peaks) <= 1:
                    single_details.append("Single dominant pitch cluster")
                if spectral_std < 500:
                    single_details.append("Consistent voice timbre")
                    
                analysis['details'] = single_details if single_details else ["Normal single-speaker characteristics"]
                analysis['explanation'] = "Voice characteristics are consistent with a single speaker"
                
    except Exception as e:
        analysis['details'] = [f"Analysis error: {str(e)}"]
        analysis['explanation'] = "Could not complete voice analysis"
        
    return analysis


def analyze_segments_for_mixed_audio(audio, model, model_type):
    """
    Analyze audio in segments to detect mixed real/AI audio.
    Returns segment-by-segment predictions and mixed audio analysis.
    """
    analysis = {
        'is_mixed': False,
        'segments': [],
        'real_segments': 0,
        'fake_segments': 0,
        'mixed_confidence': 0.0,
        'explanation': '',
        'segment_times': [],
        'segment_scores': [],
        'details': []
    }
    
    try:
        # Use 1.5 second segments with 0.5 second overlap for finer granularity
        segment_duration = 1.5  # seconds
        hop_duration = 0.5  # seconds
        segment_samples = int(SAMPLE_RATE * segment_duration)
        hop_samples = int(SAMPLE_RATE * hop_duration)
        
        if len(audio) < segment_samples:
            # Audio too short for segment analysis
            return analysis
        
        predictions = []
        times = []
        
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            segment = audio[start:start + segment_samples]
            
            # Pad segment to 3 seconds for CNN
            target_samples = SAMPLE_RATE * 3
            if len(segment) < target_samples:
                segment = np.pad(segment, (0, target_samples - len(segment)))
            
            # Compute spectrogram and predict
            spec = compute_spectrogram(segment)
            spec_input = np.expand_dims(spec, axis=0)
            pred = model.predict(spec_input, verbose=0)[0][0]
            
            predictions.append(float(pred))
            times.append((start / SAMPLE_RATE) + segment_duration / 2)  # Center time of segment
        
        predictions = np.array(predictions)
        analysis['segment_times'] = times
        analysis['segment_scores'] = predictions.tolist()
        
        # Analyze predictions
        # CNN outputs HIGH scores for REAL, LOW scores for FAKE
        # So score > 0.5 = REAL, score <= 0.5 = FAKE
        real_mask = predictions > 0.5
        fake_mask = predictions <= 0.5
        
        analysis['real_segments'] = int(np.sum(real_mask))
        analysis['fake_segments'] = int(np.sum(fake_mask))
        
        # Calculate variance and consistency
        pred_variance = np.var(predictions)
        pred_std = np.std(predictions)
        
        # Check for mixed audio indicators
        has_real = analysis['real_segments'] > 0
        has_fake = analysis['fake_segments'] > 0
        
        # Mixed if we have both real and fake segments with significant presence
        min_segment_ratio = 0.2  # At least 20% of segments
        total_segments = len(predictions)
        
        real_ratio = analysis['real_segments'] / total_segments
        fake_ratio = analysis['fake_segments'] / total_segments
        
        details = []
        
        if has_real and has_fake and min(real_ratio, fake_ratio) >= min_segment_ratio:
            analysis['is_mixed'] = True
            analysis['mixed_confidence'] = min(0.95, pred_variance * 4 + min(real_ratio, fake_ratio))
            
            # Find transition points
            transitions = []
            for i in range(1, len(predictions)):
                if (predictions[i] > 0.5) != (predictions[i-1] > 0.5):
                    transitions.append(times[i])
            
            details.append(f"Real segments: {analysis['real_segments']} ({real_ratio*100:.0f}%)")
            details.append(f"AI/Fake segments: {analysis['fake_segments']} ({fake_ratio*100:.0f}%)")
            details.append(f"Transitions detected: {len(transitions)}")
            details.append(f"Prediction variance: {pred_variance:.3f}")
            
            if len(transitions) > 0:
                analysis['explanation'] = f"⚠️ MIXED REAL/AI AUDIO DETECTED: The audio appears to contain both genuine human speech and AI-generated content. Transitions detected around {', '.join([f'{t:.1f}s' for t in transitions[:3]])}{'...' if len(transitions) > 3 else ''}."
            else:
                analysis['explanation'] = f"⚠️ MIXED REAL/AI AUDIO DETECTED: Some segments show real speech characteristics ({real_ratio*100:.0f}%) while others show AI artifacts ({fake_ratio*100:.0f}%)."
        else:
            # Not mixed - explain why
            if pred_variance < 0.05:
                details.append("Consistent predictions across all segments")
            details.append(f"Prediction std: {pred_std:.3f}")
            
            if has_real and not has_fake:
                analysis['explanation'] = "All segments consistently classified as REAL audio"
            elif has_fake and not has_real:
                analysis['explanation'] = "All segments consistently classified as AI-generated audio"
            else:
                analysis['explanation'] = f"Minor variations detected but below mixed audio threshold (minority: {min(real_ratio, fake_ratio)*100:.0f}% < 20%)"
        
        analysis['details'] = details
        
        # Create segment breakdown for visualization
        # HIGH score = REAL, LOW score = FAKE
        analysis['segments'] = [
            {
                'time': times[i],
                'score': predictions[i],
                'label': 'REAL' if predictions[i] > 0.5 else 'FAKE',
                'confidence': predictions[i] if predictions[i] > 0.5 else (1 - predictions[i])
            }
            for i in range(len(predictions))
        ]
        
    except Exception as e:
        analysis['explanation'] = f"Segment analysis error: {str(e)}"
        
    return analysis


def analyze_noise(audio):
    """Comprehensive noise analysis"""
    analysis = {
        'noise_percent': 0.0,
        'noise_type': 'clean',
        'snr_estimated': 0.0,
        'frequency_breakdown': {},
        'impact_level': 'low',
        'impact_description': '',
        'accuracy_penalty': 0.0
    }
    
    try:
        rms = librosa.feature.rms(y=audio)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        threshold = np.percentile(rms_db, 30)
        speech_mask = rms_db > threshold
        noise_mask = ~speech_mask
        
        if np.any(speech_mask) and np.any(noise_mask):
            speech_power = np.mean(rms[speech_mask] ** 2)
            noise_power = np.mean(rms[noise_mask] ** 2) + 1e-10
            snr_estimated = 10 * np.log10(speech_power / noise_power)
        else:
            snr_estimated = 30
        
        analysis['snr_estimated'] = float(snr_estimated)
        noise_percent = max(0, min(100, 50 - snr_estimated * 1.5))
        analysis['noise_percent'] = float(noise_percent)
        
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE)
        
        bands = {
            'Low (0-300Hz)': (0, 300),
            'Mid (300-3kHz)': (300, 3000),
            'High (3k-8kHz)': (3000, 8000)
        }
        
        freq_breakdown = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            if np.any(band_mask):
                band_energy = np.mean(D[band_mask, :])
                total_energy = np.mean(D) + 1e-10
                band_percent = (band_energy / total_energy) * 100
                freq_breakdown[band_name] = float(band_percent)
        
        analysis['frequency_breakdown'] = freq_breakdown
        
        low_noise = freq_breakdown.get('Low (0-300Hz)', 0)
        high_noise = freq_breakdown.get('High (3k-8kHz)', 0)
        
        if noise_percent < 10:
            analysis['noise_type'] = 'clean'
        elif low_noise > 40:
            analysis['noise_type'] = 'low_frequency_dominant'
        elif high_noise > 35:
            analysis['noise_type'] = 'high_frequency_dominant'
        else:
            analysis['noise_type'] = 'broadband'
        
        if noise_percent < 10:
            analysis['impact_level'] = 'low'
            analysis['accuracy_penalty'] = 0.0
            analysis['impact_description'] = "Clean audio - Prediction is highly reliable"
        elif noise_percent < 25:
            analysis['impact_level'] = 'low'
            analysis['accuracy_penalty'] = 0.02
            analysis['impact_description'] = "Minimal noise - Prediction accuracy: ~98%"
        elif noise_percent < 40:
            analysis['impact_level'] = 'medium'
            analysis['accuracy_penalty'] = 0.08
            analysis['impact_description'] = "Moderate noise - Accuracy may drop to ~90%"
        elif noise_percent < 55:
            analysis['impact_level'] = 'high'
            analysis['accuracy_penalty'] = 0.15
            analysis['impact_description'] = "High noise - Accuracy: ~85%. Noise may mask artifacts"
        else:
            analysis['impact_level'] = 'very_high'
            analysis['accuracy_penalty'] = 0.25
            analysis['impact_description'] = "Very high noise - Accuracy: ~75%. Prediction may be unreliable!"
            
    except Exception as e:
        analysis['impact_description'] = f"Analysis error: {str(e)}"
        
    return analysis


def predict_cnn(audio):
    """Run CNN prediction"""
    global loaded_model
    
    target_samples = SAMPLE_RATE * 3
    
    if len(audio) <= target_samples:
        if len(audio) < target_samples:
            audio_chunk = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio_chunk = audio
        spectrogram = compute_spectrogram(audio_chunk)
        spec_input = np.expand_dims(spectrogram, axis=0)
        prediction = loaded_model.predict(spec_input, verbose=0)[0][0]
    else:
        chunk_size = target_samples
        hop_size = int(target_samples * 0.5)
        predictions = []
        
        for start in range(0, len(audio) - chunk_size + 1, hop_size):
            chunk = audio[start:start + chunk_size]
            spec = compute_spectrogram(chunk)
            spec_input = np.expand_dims(spec, axis=0)
            pred = loaded_model.predict(spec_input, verbose=0)[0][0]
            predictions.append(pred)
        
        prediction = np.mean(predictions)
        spectrogram = compute_spectrogram(audio[-chunk_size:])
    
    # CNN outputs HIGH scores for REAL, LOW scores for FAKE
    # So score > 0.5 = REAL, score <= 0.5 = FAKE
    is_real = prediction > 0.5
    confidence = prediction if is_real else (1 - prediction)
    
    return {
        'prediction': 'REAL' if is_real else 'FAKE',
        'confidence': float(confidence),
        'raw_score': float(prediction),
        'spectrogram': spectrogram
    }


def predict_traditional(audio):
    """Run traditional model prediction"""
    global loaded_model
    
    mfcc = compute_mfcc(audio)
    mfcc_input = mfcc.reshape(1, -1)
    
    if hasattr(loaded_model, 'predict_proba'):
        proba = loaded_model.predict_proba(mfcc_input)[0]
        prediction = loaded_model.predict(mfcc_input)[0]
        confidence = max(proba)
    else:
        prediction = loaded_model.predict(mfcc_input)[0]
        confidence = 0.9
    
    is_real = prediction == 1
    
    return {
        'prediction': 'REAL' if is_real else 'FAKE',
        'confidence': float(confidence),
        'raw_score': float(prediction),
        'mfcc': mfcc.tolist()
    }


def generate_visualizations(audio, spectrogram, prediction, confidence):
    """Generate all visualization plots as base64 images"""
    visualizations = {}
    
    plt.style.use('dark_background')
    
    # Use bigger figure size for better visibility
    fig_size = (10, 5)  # Bigger figures
    
    # 1. Waveform
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    time = np.linspace(0, len(audio)/SAMPLE_RATE, len(audio))
    ax.plot(time, audio, color='#4cc9f0', linewidth=0.5, alpha=0.8)
    ax.fill_between(time, audio, alpha=0.3, color='#4cc9f0')
    ax.set_xlabel('Time (s)', color='#888', fontsize=9)
    ax.set_ylabel('Amplitude', color='#888', fontsize=9)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['waveform'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 2. Spectrogram
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    spec_display = spectrogram[:, :, 0].T
    im = ax.imshow(spec_display, aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel('Time →', color='#888', fontsize=11)
    ax.set_ylabel('Mel Frequency →', color='#888', fontsize=11)
    ax.tick_params(colors='#888')
    
    # Add detection markers
    h, w = spec_display.shape
    grid_h, grid_w = 4, 8
    cell_h, cell_w = h // grid_h, w // grid_w
    detection_regions = []
    for i in range(grid_h):
        for j in range(grid_w):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = spec_display[y1:y2, x1:x2]
            score = np.mean(cell) * 0.5 + np.var(cell) * 0.5
            detection_regions.append((score, x1, y1, x2, y2))
    detection_regions.sort(reverse=True)
    marker_color = '#06d6a0' if prediction == 'REAL' else '#ef476f'
    for idx, (score, x1, y1, x2, y2) in enumerate(detection_regions[:3]):
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=marker_color,
                         facecolor=marker_color, alpha=0.25)
        ax.add_patch(rect)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(cx, cy, str(idx+1), color='white', fontsize=12, ha='center', va='center',
                fontweight='bold', bbox=dict(boxstyle='circle', facecolor=marker_color, alpha=0.8))
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['spectrogram'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 3. Pitch Contour
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        pitch_values = []
        pitch_times = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                pitch_times.append(t * 512 / SAMPLE_RATE)
        if pitch_values:
            ax.scatter(pitch_times, pitch_values, c='#ffd166', s=4, alpha=0.7)
            ax.plot(pitch_times, pitch_values, color='#ffd166', linewidth=1, alpha=0.5)
    except:
        pass
    ax.set_xlabel('Time (s)', color='#888', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', color='#888', fontsize=11)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['pitch'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 4. Saliency Heatmap
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    grad_x = ndimage.sobel(spec_display, axis=1)
    grad_y = ndimage.sobel(spec_display, axis=0)
    saliency = np.sqrt(grad_x**2 + grad_y**2)
    saliency = ndimage.gaussian_filter(saliency, sigma=5)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    im = ax.imshow(saliency, aspect='auto', origin='lower', cmap='hot')
    ax.set_xlabel('Time →', color='#888', fontsize=11)
    ax.set_ylabel('Frequency →', color='#888', fontsize=11)
    ax.tick_params(colors='#888')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['saliency'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 5. Spectral Centroid
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
    frames = range(len(spectral_centroids))
    t_frames = librosa.frames_to_time(frames, sr=SAMPLE_RATE)
    ax.plot(t_frames, spectral_centroids, color='#4361ee', linewidth=1.5)
    ax.fill_between(t_frames, spectral_centroids, alpha=0.3, color='#4361ee')
    ax.axhline(y=np.mean(spectral_centroids), color='#ef476f', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time (s)', color='#888', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', color='#888', fontsize=11)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['centroid'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 6. Zero Crossing Rate
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    t_zcr = librosa.frames_to_time(range(len(zcr)), sr=SAMPLE_RATE)
    ax.plot(t_zcr, zcr, color='#06d6a0', linewidth=1.5)
    ax.fill_between(t_zcr, zcr, alpha=0.3, color='#06d6a0')
    ax.set_xlabel('Time (s)', color='#888', fontsize=11)
    ax.set_ylabel('Rate', color='#888', fontsize=11)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['zcr'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 7. RMS Energy
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    rms = librosa.feature.rms(y=audio)[0]
    t_rms = librosa.frames_to_time(range(len(rms)), sr=SAMPLE_RATE)
    ax.plot(t_rms, rms, color='#ef476f', linewidth=1.5)
    ax.fill_between(t_rms, rms, alpha=0.3, color='#ef476f')
    ax.axhline(y=np.mean(rms), color='#ffd166', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time (s)', color='#888', fontsize=11)
    ax.set_ylabel('Energy', color='#888', fontsize=11)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    plt.tight_layout()
    visualizations['rms'] = fig_to_base64(fig)
    plt.close(fig)
    
    return visualizations


def generate_voice_analysis_figure(voice_analysis, audio):
    """Generate visualization explaining voice analysis detection"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#1a1a2e')
    plt.style.use('dark_background')
    
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3e')
    
    # 1. Pitch over time
    ax1 = axes[0, 0]
    pitch_data = voice_analysis.get('pitch_data', {})
    if pitch_data and 'times' in pitch_data and 'values' in pitch_data:
        times = pitch_data['times']
        values = pitch_data['values']
        ax1.scatter(times, values, c='#ffd166', s=8, alpha=0.6)
        ax1.plot(times, values, color='#ffd166', linewidth=0.5, alpha=0.3)
        
        # Add mean line
        mean_pitch = np.mean(values)
        ax1.axhline(y=mean_pitch, color='#4cc9f0', linestyle='--', linewidth=2, label=f'Mean: {mean_pitch:.0f} Hz')
        
        # Add std bands
        std_pitch = np.std(values)
        ax1.axhspan(mean_pitch - std_pitch, mean_pitch + std_pitch, alpha=0.2, color='#4cc9f0', label=f'±1 STD: {std_pitch:.0f} Hz')
        
        ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlabel('Time (s)', color='#888', fontsize=10)
    ax1.set_ylabel('Pitch (Hz)', color='#888', fontsize=10)
    ax1.set_title('Pitch Contour Over Time', color='#fff', fontsize=11, fontweight='bold')
    ax1.tick_params(colors='#888')
    
    # 2. Pitch histogram
    ax2 = axes[0, 1]
    if pitch_data and 'values' in pitch_data:
        values = pitch_data['values']
        n, bins, patches = ax2.hist(values, bins=30, color='#4361ee', alpha=0.7, edgecolor='#2a2a3e')
        
        # Color patches by frequency
        for i, p in enumerate(patches):
            if n[i] > np.max(n) * 0.5:
                p.set_facecolor('#06d6a0')  # High frequency = green
            elif n[i] > np.max(n) * 0.25:
                p.set_facecolor('#ffd166')  # Medium = yellow
    ax2.set_xlabel('Pitch (Hz)', color='#888', fontsize=10)
    ax2.set_ylabel('Frequency', color='#888', fontsize=10)
    ax2.set_title('Pitch Distribution (Clusters = Multiple Voices)', color='#fff', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='#888')
    
    # 3. Spectral centroid
    ax3 = axes[1, 0]
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
    frames = range(len(spectral_centroids))
    t_frames = librosa.frames_to_time(frames, sr=SAMPLE_RATE)
    ax3.plot(t_frames, spectral_centroids, color='#4cc9f0', linewidth=1)
    ax3.fill_between(t_frames, spectral_centroids, alpha=0.3, color='#4cc9f0')
    
    # Add mean and std
    mean_centroid = np.mean(spectral_centroids)
    std_centroid = np.std(spectral_centroids)
    ax3.axhline(y=mean_centroid, color='#ef476f', linestyle='--', linewidth=2, label=f'Mean: {mean_centroid:.0f} Hz')
    ax3.axhspan(mean_centroid - std_centroid, mean_centroid + std_centroid, alpha=0.15, color='#ef476f')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlabel('Time (s)', color='#888', fontsize=10)
    ax3.set_ylabel('Frequency (Hz)', color='#888', fontsize=10)
    ax3.set_title(f'Spectral Centroid (σ={std_centroid:.0f} Hz)', color='#fff', fontsize=11, fontweight='bold')
    ax3.tick_params(colors='#888')
    
    # 4. Analysis summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    source_type = voice_analysis.get('source_type', 'unknown')
    num_sources = voice_analysis.get('num_sources', 1)
    confidence = voice_analysis.get('confidence', 0)
    details = voice_analysis.get('details', [])
    explanation = voice_analysis.get('explanation', '')
    
    # Create summary text
    if source_type == 'single_voice':
        verdict_color = '#06d6a0'
        verdict_text = '[OK] SINGLE VOICE'
    else:
        verdict_color = '#ffd166'
        verdict_text = '[!] MULTIPLE VOICES'
    
    summary_text = f"""
    Voice Analysis Result
    ─────────────────────
    
    Detection: {verdict_text}
    Confidence: {confidence*100:.1f}%
    
    Indicators:
    """
    for d in details[:4]:
        summary_text += f"\n    • {d}"
    
    summary_text += f"""
    
    Explanation:
    {explanation[:100]}{'...' if len(explanation) > 100 else ''}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', color='#fff', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#2a2a3e', alpha=0.8))
    
    # Add verdict badge
    ax4.text(0.5, 0.15, verdict_text, transform=ax4.transAxes, fontsize=16,
             ha='center', color=verdict_color, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor=verdict_color, linewidth=2))
    
    plt.tight_layout()
    return fig_to_base64(fig)


def generate_segment_analysis_figure(segment_analysis):
    """Generate visualization for segment-by-segment prediction analysis"""
    # Create figure with simple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#1a1a2e')
    plt.style.use('dark_background')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3e')
    
    times = segment_analysis.get('segment_times', [])
    scores = segment_analysis.get('segment_scores', [])
    
    if not times or not scores:
        # No segment data - show message
        ax1.text(0.5, 0.5, 'Not enough audio for segment analysis\n(Need at least 1.5 seconds)', 
                 ha='center', va='center', fontsize=12, color='#888',
                 transform=ax1.transAxes)
        ax1.axis('off')
        ax2.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        result = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return result
    
    times = np.array(times)
    scores = np.array(scores)
    
    # TOP PLOT: Bar chart of segments
    # HIGH score = REAL (green), LOW score = FAKE (red)
    colors = ['#06d6a0' if s > 0.5 else '#ef476f' for s in scores]
    realness = scores  # Score IS realness (higher = more real)
    
    bar_width = 0.4
    if len(times) > 1:
        bar_width = (times[1] - times[0]) * 0.8
    
    ax1.bar(times, realness, width=bar_width, color=colors, alpha=0.8)
    ax1.axhline(y=0.5, color='#ffd166', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time (seconds)', color='#aaa', fontsize=10)
    ax1.set_ylabel('Realness Score', color='#aaa', fontsize=10)
    ax1.set_title('Segment-by-Segment Analysis', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='#aaa')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='#06d6a0', label='REAL'),
        Patch(facecolor='#ef476f', label='FAKE/AI'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', fontsize=9)
    
    # BOTTOM PLOT: Summary stats as text
    ax2.axis('off')
    
    real_count = segment_analysis.get('real_segments', 0)
    fake_count = segment_analysis.get('fake_segments', 0)
    total = real_count + fake_count
    is_mixed = segment_analysis.get('is_mixed', False)
    
    if is_mixed:
        verdict = "[!] MIXED REAL/AI DETECTED"
        verdict_color = '#ffd166'
    elif real_count > fake_count:
        verdict = "[OK] CONSISTENT - REAL"
        verdict_color = '#06d6a0'
    else:
        verdict = "[OK] CONSISTENT - FAKE"
        verdict_color = '#ef476f'
    
    summary = f"""
    {verdict}
    
    Total Segments Analyzed: {total}
    Real Segments: {real_count} ({real_count/total*100:.0f}% of audio)
    Fake Segments: {fake_count} ({fake_count/total*100:.0f}% of audio)
    """
    
    ax2.text(0.5, 0.7, summary, transform=ax2.transAxes, fontsize=11,
             ha='center', va='center', color='white', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a3e', alpha=0.9))
    
    ax2.text(0.5, 0.15, verdict, transform=ax2.transAxes, fontsize=14,
             ha='center', va='center', color=verdict_color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    result = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return result


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    result = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return result


def _load_audio_from_request(req):
    """Load audio from multipart upload or base64 JSON.

    Returns (audio_np_float32_mono_16k, temp_files_list).
    """
    temp_files = []

    if 'audio' in req.files:
        audio_file = req.files['audio']
        original_ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        temp_input = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex[:8]}{original_ext}")
        temp_wav = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.wav")
        temp_files.extend([temp_input, temp_wav])
        audio_file.save(temp_input)

        # Avoid PySoundFile warnings for formats libsndfile doesn't handle well (e.g., webm/m4a).
        ext = original_ext.lower()
        prefer_direct = ext in ['.wav', '.flac']

        if prefer_direct:
            try:
                audio, _ = librosa.load(temp_input, sr=SAMPLE_RATE, mono=True)
                return audio.astype(np.float32), temp_files
            except Exception:
                pass

        if convert_audio_to_wav(temp_input, temp_wav):
            audio, _ = librosa.load(temp_wav, sr=SAMPLE_RATE, mono=True)
            return audio.astype(np.float32), temp_files

        # Final fallback: let librosa/audioread try directly.
        audio, _ = librosa.load(temp_input, sr=SAMPLE_RATE, mono=True)

        return audio.astype(np.float32), temp_files

    if req.is_json and req.json and 'audio_data' in req.json:
        audio_b64 = req.json['audio_data']
        audio_bytes = base64.b64decode(audio_b64)

        temp_input = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.webm")
        temp_wav = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.wav")
        temp_files.extend([temp_input, temp_wav])

        with open(temp_input, 'wb') as f:
            f.write(audio_bytes)

        # Recorded audio is typically webm; prefer conversion to avoid warnings and improve consistency.
        if convert_audio_to_wav(temp_input, temp_wav):
            audio, _ = librosa.load(temp_wav, sr=SAMPLE_RATE, mono=True)
            return audio.astype(np.float32), temp_files

        # Fallback: let librosa/audioread try directly.
        audio, _ = librosa.load(temp_input, sr=SAMPLE_RATE, mono=True)

        return audio.astype(np.float32), temp_files

    raise RuntimeError('No audio data provided')


def _apply_requested_noise(req, audio, temp_files):
    """Apply SNR simulation and/or custom noise mixing based on request."""

    # Apply SNR if specified
    snr_level = req.form.get('snr') or (req.json.get('snr') if req.is_json and req.json else None)
    if snr_level and snr_level != 'None':
        snr_db = int(str(snr_level).replace(' dB', ''))
        signal_power = np.mean(audio ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        audio = (audio + noise).astype(np.float32)

    # Apply custom noise if provided
    if 'custom_noise' in req.files:
        noise_file = req.files['custom_noise']
        noise_mix_level = float(req.form.get('noise_mix_level', 30)) / 100.0

        noise_ext = os.path.splitext(noise_file.filename)[1] or '.wav'
        temp_noise_input = os.path.join(UPLOAD_FOLDER, f"temp_noise_{uuid.uuid4().hex[:8]}{noise_ext}")
        temp_noise_wav = os.path.join(UPLOAD_FOLDER, f"temp_noise_{uuid.uuid4().hex[:8]}.wav")
        temp_files.extend([temp_noise_input, temp_noise_wav])
        noise_file.save(temp_noise_input)

        try:
            custom_noise, _ = librosa.load(temp_noise_input, sr=SAMPLE_RATE, mono=True)
        except Exception:
            if convert_audio_to_wav(temp_noise_input, temp_noise_wav):
                custom_noise, _ = librosa.load(temp_noise_wav, sr=SAMPLE_RATE, mono=True)
            else:
                custom_noise = None

        if custom_noise is not None:
            max_noise_samples = SAMPLE_RATE * 10
            if len(custom_noise) > max_noise_samples:
                custom_noise = custom_noise[:max_noise_samples]

            if len(custom_noise) < len(audio):
                repeats = int(np.ceil(len(audio) / len(custom_noise)))
                custom_noise = np.tile(custom_noise, repeats)[:len(audio)]
            else:
                custom_noise = custom_noise[:len(audio)]

            custom_noise = custom_noise / (np.max(np.abs(custom_noise)) + 1e-8)
            audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
            audio = (audio_normalized * (1 - noise_mix_level) + custom_noise * noise_mix_level).astype(np.float32)

    return audio


def _get_hf_resnet_v2_pipeline():
    """Lazy-load Hugging Face audio-classification pipeline (cached)."""
    global _hf_resnet_v2_pipeline
    if _hf_resnet_v2_pipeline is not None:
        return _hf_resnet_v2_pipeline

    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError(
            "ResNet v2 backend dependencies are missing. Install: pip install transformers torch torchaudio"
        ) from e

    _hf_resnet_v2_pipeline = pipeline(
        task="audio-classification",
        model=HF_RESNET_V2_MODEL_ID,
    )
    return _hf_resnet_v2_pipeline


def predict_resnet_v2(audio):
    """Run backup model prediction using HF pipeline.

    Returns: {prediction: 'REAL'|'FAKE', confidence: float, raw_score: float, model_label: str, scores: list}
    """
    pipe = _get_hf_resnet_v2_pipeline()

    outputs = pipe({"array": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE})
    if isinstance(outputs, dict):
        outputs = [outputs]

    if not outputs:
        raise RuntimeError("ResNet v2 returned no outputs")

    # Ensure sorted by score desc
    outputs = sorted(outputs, key=lambda x: float(x.get('score', 0.0)), reverse=True)
    top = outputs[0]
    label = str(top.get('label', '')).lower()
    score = float(top.get('score', 0.0))

    # Robust mapping from label -> verdict
    if any(k in label for k in ['fake', 'spoof', 'ai', 'deepfake']):
        pred = 'FAKE'
    elif any(k in label for k in ['real', 'bonafide', 'bona fide', 'genuine']):
        pred = 'REAL'
    else:
        # Fallback: if two classes exist, try to infer by the *other* label
        pred = 'FAKE'
        if len(outputs) >= 2:
            other_label = str(outputs[1].get('label', '')).lower()
            if any(k in other_label for k in ['fake', 'spoof', 'ai', 'deepfake']):
                pred = 'REAL'
            elif any(k in other_label for k in ['real', 'bonafide', 'bona fide', 'genuine']):
                pred = 'FAKE'

    return {
        'prediction': pred,
        'confidence': float(score),
        'raw_score': float(score),
        'model_label': str(top.get('label', '')),
        'scores': [{'label': str(o.get('label', '')), 'score': float(o.get('score', 0.0))} for o in outputs]
    }


# ==========================================
# 🌐 ROUTES
# ==========================================

@app.route('/api/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': loaded_model is not None,
        'gemini_available': gemini_service.is_available() if gemini_service else False
    })


@app.route('/api/model/load', methods=['POST'])
def api_load_model():
    """Load a model"""
    global model_path
    
    data = request.json
    mtype = data.get('type', 'CNN')
    
    # Check if model file was uploaded
    if 'path' in data and data['path']:
        path = data['path']
    else:
        # Try to find default model
        path = find_default_model()
        if not path:
            return jsonify({'success': False, 'error': 'No model found. Please upload a model file.'})
    
    success, result = load_model_file(path, mtype)
    
    if success:
        return jsonify({'success': True, 'model_name': result})
    else:
        return jsonify({'success': False, 'error': result})


@app.route('/api/model/upload', methods=['POST'])
def api_upload_model():
    """Upload and load a model file"""
    if 'model' not in request.files:
        return jsonify({'success': False, 'error': 'No model file provided'})
    
    file = request.files['model']
    mtype = request.form.get('type', 'CNN')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save the file
    filename = f"uploaded_model_{uuid.uuid4().hex[:8]}{os.path.splitext(file.filename)[1]}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Load the model
    success, result = load_model_file(filepath, mtype)
    
    if success:
        return jsonify({'success': True, 'model_name': result})
    else:
        os.remove(filepath)
        return jsonify({'success': False, 'error': result})


@app.route('/api/model/status')
def api_model_status():
    """Get current model status"""
    if loaded_model is not None:
        return jsonify({
            'loaded': True,
            'type': model_type,
            'name': os.path.basename(model_path) if model_path else 'Unknown'
        })
    return jsonify({'loaded': False})


def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        # Try using ffmpeg
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ar', str(SAMPLE_RATE), '-ac', '1', output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback: try pydub if available
    try:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(input_path)
        audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio_segment.export(output_path, format='wav')
        return True
    except Exception:
        pass
    
    return False


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Analyze uploaded audio"""
    global loaded_model, model_type
    
    if loaded_model is None:
        return jsonify({'success': False, 'error': 'No model loaded'})
    
    temp_files = []  # Track temp files for cleanup
    
    try:
        audio, temp_files = _load_audio_from_request(request)
        audio = _apply_requested_noise(request, audio, temp_files)
    
        # Run analysis
        # Voice analysis
        voice_analysis = analyze_voices(audio)
        
        # Noise analysis
        noise_analysis = analyze_noise(audio)
        
        # Prediction
        if model_type == "CNN":
            result = predict_cnn(audio)
            spectrogram = result['spectrogram']
            
            # Segment analysis for mixed audio detection (CNN only)
            segment_analysis = analyze_segments_for_mixed_audio(audio, loaded_model, model_type)
        else:
            result = predict_traditional(audio)
            # Generate spectrogram for visualization anyway
            target_samples = SAMPLE_RATE * 3
            if len(audio) < target_samples:
                audio_padded = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio_padded = audio[:target_samples]
            spectrogram = compute_spectrogram(audio_padded)
            segment_analysis = {'is_mixed': False, 'segments': [], 'explanation': 'Segment analysis only available for CNN models'}
        
        # Generate visualizations
        visualizations = generate_visualizations(
            audio, spectrogram, result['prediction'], result['confidence']
        )
        
        # Generate voice analysis figure
        visualizations['voice_analysis_figure'] = generate_voice_analysis_figure(voice_analysis, audio)
        
        # Generate segment analysis figure (for mixed audio detection) - always generate
        try:
            visualizations['segment_analysis_figure'] = generate_segment_analysis_figure(segment_analysis)
        except Exception as e:
            print(f"Error generating segment analysis figure: {e}")
            visualizations['segment_analysis_figure'] = None
        
        # Calculate adjusted confidence
        adjusted_confidence = result['confidence'] * (1 - noise_analysis['accuracy_penalty'])
        
        # If mixed audio detected, add warning to result
        mixed_audio_warning = None
        if segment_analysis.get('is_mixed'):
            mixed_audio_warning = segment_analysis.get('explanation', 'Mixed real/AI audio detected')
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'adjusted_confidence': adjusted_confidence,
            'raw_score': result['raw_score'],
            'voice_analysis': voice_analysis,
            'noise_analysis': noise_analysis,
            'segment_analysis': segment_analysis,
            'mixed_audio_warning': mixed_audio_warning,
            'visualizations': visualizations,
            'audio_duration': len(audio) / SAMPLE_RATE
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass


@app.route('/api/analyze_resnet_v2', methods=['POST'])
def api_analyze_resnet_v2():
    """Analyze audio using the backup "ResNet v2" model (HF pipeline)."""

    temp_files = []

    try:
        audio, temp_files = _load_audio_from_request(request)
        audio = _apply_requested_noise(request, audio, temp_files)

        # Voice + noise analysis are the same utilities
        voice_analysis = analyze_voices(audio)
        noise_analysis = analyze_noise(audio)

        # Prediction via HF model
        pred = predict_resnet_v2(audio)

        # Generate spectrogram for visualization (pad/trim to 3s)
        target_samples = SAMPLE_RATE * 3
        if len(audio) < target_samples:
            audio_padded = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio_padded = audio[:target_samples]
        spectrogram = compute_spectrogram(audio_padded)

        visualizations = generate_visualizations(audio, spectrogram, pred['prediction'], pred['confidence'])
        visualizations['voice_analysis_figure'] = generate_voice_analysis_figure(voice_analysis, audio)

        # Segment analysis is only available for the CNN model path
        segment_analysis = {
            'is_mixed': False,
            'segments': [],
            'explanation': 'Segment analysis only available for CNN models'
        }
        visualizations['segment_analysis_figure'] = None

        adjusted_confidence = float(pred['confidence']) * (1 - noise_analysis.get('accuracy_penalty', 0.0))

        return jsonify({
            'success': True,
            'engine': 'resnet_v2',
            'prediction': pred['prediction'],
            'confidence': float(pred['confidence']),
            'adjusted_confidence': float(adjusted_confidence),
            'raw_score': float(pred['raw_score']),
            'model_label': pred.get('model_label'),
            'scores': pred.get('scores'),
            'voice_analysis': voice_analysis,
            'noise_analysis': noise_analysis,
            'segment_analysis': segment_analysis,
            'mixed_audio_warning': None,
            'visualizations': visualizations,
            'audio_duration': len(audio) / SAMPLE_RATE
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass


@app.route('/api/gradcam', methods=['POST'])
def api_gradcam():
    """Generate Grad-CAM visualization for CNN model."""
    global loaded_model, model_type
    
    if loaded_model is None:
        return jsonify({'success': False, 'error': 'No model loaded'})
    
    if model_type != "CNN":
        return jsonify({'success': False, 'error': 'Grad-CAM only available for CNN models', 'available': False})
    
    temp_files = []
    
    try:
        audio, temp_files = _load_audio_from_request(request)
        audio = _apply_requested_noise(request, audio, temp_files)
        
        # Prepare input
        target_samples = SAMPLE_RATE * 3
        if len(audio) < target_samples:
            audio_chunk = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio_chunk = audio[:target_samples]
        
        spectrogram = compute_spectrogram(audio_chunk)
        spec_input = np.expand_dims(spectrogram, axis=0)
        spec_tensor = tf.constant(spec_input, dtype=tf.float32)
        
        # For ResNet models with nested architecture, we need a different approach
        # Use gradient-based saliency as a fallback that works with any model
        
        try:
            # Try to find ResNet50 submodel first
            resnet_submodel = None
            last_conv_layer = None
            
            for layer in loaded_model.layers:
                if 'resnet' in layer.name.lower():
                    resnet_submodel = layer
                    break
            
            if resnet_submodel is not None:
                # Find last conv layer in the ResNet submodel
                for layer in reversed(resnet_submodel.layers):
                    if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
                        last_conv_layer = layer
                        break
            
            if last_conv_layer is None:
                # Fallback: find any conv layer in main model
                for layer in reversed(loaded_model.layers):
                    if 'conv' in layer.name.lower():
                        last_conv_layer = layer
                        break
            
            # Use gradient-based saliency map instead of Grad-CAM for complex architectures
            with tf.GradientTape() as tape:
                tape.watch(spec_tensor)
                predictions = loaded_model(spec_tensor, training=False)
                if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                    class_output = predictions[:, 0]
                else:
                    class_output = tf.squeeze(predictions)
            
            # Compute gradients with respect to input
            grads = tape.gradient(class_output, spec_tensor)
            
            if grads is None:
                raise ValueError("Could not compute gradients")
            
            # Create saliency map from gradients
            saliency = tf.abs(grads)
            saliency = tf.reduce_mean(saliency, axis=-1)  # Average over channels
            saliency = saliency[0].numpy()  # Remove batch dimension
            
            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            # Apply Gaussian smoothing for better visualization
            from scipy.ndimage import gaussian_filter
            saliency_smooth = gaussian_filter(saliency, sigma=3)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            # Show spectrogram
            spec_display = spectrogram[:, :, 0].T
            ax.imshow(spec_display, aspect='auto', origin='lower', cmap='magma', alpha=0.7)
            
            # Overlay saliency heatmap
            ax.imshow(saliency_smooth.T, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
            
            ax.set_xlabel('Time →', color='#888', fontsize=11)
            ax.set_ylabel('Mel Frequency →', color='#888', fontsize=11)
            ax.set_title('Gradient Saliency Map: Model Attention', color='#fff', fontsize=12, fontweight='bold')
            ax.tick_params(colors='#888')
            
            for spine in ax.spines.values():
                spine.set_color('#2a2a3e')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='jet')
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label='Attention')
            cbar.ax.yaxis.label.set_color('#888')
            cbar.ax.tick_params(colors='#888')
            
            plt.tight_layout()
            gradcam_base64 = fig_to_base64(fig)
            plt.close(fig)
            
            return jsonify({
                'success': True,
                'available': True,
                'gradcam': gradcam_base64,
                'method': 'gradient_saliency'
            })
            
        except Exception as inner_e:
            # Ultimate fallback: activation-based heatmap
            import traceback
            print(f"Saliency fallback failed: {inner_e}")
            print(traceback.format_exc())
            
            # Just show which parts of spectrogram have highest energy
            spec_energy = np.abs(spectrogram[:, :, 0])
            spec_energy = (spec_energy - spec_energy.min()) / (spec_energy.max() - spec_energy.min() + 1e-8)
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            ax.imshow(spec_energy.T, aspect='auto', origin='lower', cmap='magma')
            ax.set_xlabel('Time →', color='#888', fontsize=11)
            ax.set_ylabel('Mel Frequency →', color='#888', fontsize=11)
            ax.set_title('Spectrogram Energy Distribution', color='#fff', fontsize=12, fontweight='bold')
            ax.tick_params(colors='#888')
            
            for spine in ax.spines.values():
                spine.set_color('#2a2a3e')
            
            plt.tight_layout()
            gradcam_base64 = fig_to_base64(fig)
            plt.close(fig)
            
            return jsonify({
                'success': True,
                'available': True,
                'gradcam': gradcam_base64,
                'method': 'energy_fallback',
                'note': 'Using energy-based visualization due to model architecture'
            })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})
    
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


@app.route('/api/shap', methods=['POST'])
def api_shap():
    """Compute SHAP feature importance for Traditional ML models."""
    global loaded_model, model_type
    
    if loaded_model is None:
        return jsonify({'success': False, 'error': 'No model loaded', 'available': False})
    
    temp_files = []
    
    try:
        audio, temp_files = _load_audio_from_request(request)
        
        # Compute MFCC features
        mfcc = compute_mfcc(audio)
        mfcc_input = mfcc.reshape(1, -1)
        
        # Feature names for MFCC
        feature_names = []
        for i in range(40):
            feature_names.append(f'MFCC_{i+1}_mean')
        for i in range(40):
            feature_names.append(f'MFCC_{i+1}_std')
        
        if model_type == "CNN":
            # For CNN, we can compute a simple feature importance based on spectrogram regions
            # This is a simplified "pseudo-SHAP" for CNN
            target_samples = SAMPLE_RATE * 3
            if len(audio) < target_samples:
                audio_chunk = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio_chunk = audio[:target_samples]
            
            spectrogram = compute_spectrogram(audio_chunk)
            spec_input = np.expand_dims(spectrogram, axis=0)
            base_pred = loaded_model.predict(spec_input, verbose=0)[0][0]
            
            # Compute importance by masking frequency bands
            importance_scores = []
            band_names = ['Low (0-1kHz)', 'Low-Mid (1-2kHz)', 'Mid (2-4kHz)', 'High-Mid (4-6kHz)', 'High (6-8kHz)']
            
            for i, band_name in enumerate(band_names):
                # Create masked spectrogram
                masked_spec = spectrogram.copy()
                band_start = int(i * 224 / 5)
                band_end = int((i + 1) * 224 / 5)
                masked_spec[band_start:band_end, :, :] = 0
                
                masked_input = np.expand_dims(masked_spec, axis=0)
                masked_pred = loaded_model.predict(masked_input, verbose=0)[0][0]
                
                # Importance = how much prediction changed when band was removed
                importance = abs(base_pred - masked_pred)
                importance_scores.append({
                    'name': band_name,
                    'value': float(importance)
                })
            
            # Sort by absolute importance
            importance_scores.sort(key=lambda x: abs(x['value']), reverse=True)
            
            return jsonify({
                'success': True,
                'available': True,
                'method': 'frequency_band_masking',
                'top_features': importance_scores,
                'note': 'Showing frequency band importance for CNN model'
            })
        
        else:
            # Traditional ML - try to use SHAP
            try:
                import shap
                
                # Create explainer
                if hasattr(loaded_model, 'predict_proba'):
                    explainer = shap.Explainer(loaded_model.predict_proba, mfcc_input)
                else:
                    explainer = shap.Explainer(loaded_model.predict, mfcc_input)
                
                shap_values = explainer(mfcc_input)
                
                # Get feature importance
                if hasattr(shap_values, 'values'):
                    values = shap_values.values[0]
                    if len(values.shape) > 1:
                        values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
                else:
                    values = np.abs(shap_values[0])
                
                # Create list of features with importance
                feature_importance = []
                for i, (name, val) in enumerate(zip(feature_names, values)):
                    feature_importance.append({
                        'name': name,
                        'value': float(val)
                    })
                
                # Sort by absolute importance and get top 10
                feature_importance.sort(key=lambda x: abs(x['value']), reverse=True)
                top_features = feature_importance[:10]
                
                return jsonify({
                    'success': True,
                    'available': True,
                    'method': 'shap',
                    'top_features': top_features
                })
                
            except ImportError:
                # SHAP not installed - use permutation importance instead
                if hasattr(loaded_model, 'feature_importances_'):
                    # Tree-based models have feature_importances_
                    importances = loaded_model.feature_importances_
                    feature_importance = []
                    for i, (name, val) in enumerate(zip(feature_names, importances)):
                        feature_importance.append({
                            'name': name,
                            'value': float(val)
                        })
                    feature_importance.sort(key=lambda x: abs(x['value']), reverse=True)
                    
                    return jsonify({
                        'success': True,
                        'available': True,
                        'method': 'feature_importances',
                        'top_features': feature_importance[:10]
                    })
                else:
                    # Compute simple perturbation-based importance
                    base_pred = loaded_model.predict(mfcc_input)[0]
                    importance_scores = []
                    
                    for i, name in enumerate(feature_names[:20]):  # Only first 20 for speed
                        perturbed = mfcc_input.copy()
                        perturbed[0, i] = 0  # Zero out feature
                        new_pred = loaded_model.predict(perturbed)[0]
                        importance = abs(base_pred - new_pred)
                        if hasattr(importance, '__len__'):
                            importance = importance[0] if len(importance) > 0 else 0
                        importance_scores.append({
                            'name': name,
                            'value': float(importance)
                        })
                    
                    importance_scores.sort(key=lambda x: abs(x['value']), reverse=True)
                    
                    return jsonify({
                        'success': True,
                        'available': True,
                        'method': 'perturbation',
                        'top_features': importance_scores[:10]
                    })
    
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc(), 'available': False})
    
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


@app.route('/api/explain_visualization', methods=['POST'])
def api_explain_visualization():
    """Generate AI explanation for a specific visualization using Gemini."""
    global gemini_service
    
    try:
        data = request.get_json()
        viz_type = data.get('viz_type', 'unknown')
        prediction = data.get('prediction', 'unknown')
        confidence = data.get('confidence', 0)
        
        # Additional context from analysis
        context = {
            'audio_duration': data.get('audio_duration', 0),
            'pitch_std': data.get('pitch_std', 0),
            'spectral_variation': data.get('spectral_variation', 0),
            'snr': data.get('snr', 0)
        }
        
        # Initialize Gemini if not already done
        if gemini_service is None and GEMINI_API_KEY:
            gemini_service = get_gemini_service(GEMINI_API_KEY)
        
        # Use Gemini service if available
        if gemini_service and gemini_service.is_available():
            result = gemini_service.explain_visualization(viz_type, prediction, confidence, context)
            return jsonify(result)
        
        # Fallback to static explanations
        static_explanations = {
            'waveform': f"This waveform shows how the sound's loudness changes over time. The wavy pattern represents the audio signal - peaks are loud moments, valleys are quiet moments. This audio was classified as {prediction} with {confidence:.1%} confidence. Natural human speech typically shows varied, organic patterns, while synthetic audio may appear unnaturally smooth or repetitive.",
            
            'spectrogram': f"This colorful 'heat map' shows sound frequencies over time. Brighter colors indicate louder sounds at those frequencies. The horizontal axis is time, vertical is pitch (low to high). This audio was classified as {prediction} ({confidence:.1%} confidence). Real voices create natural harmonic patterns, while AI-generated audio may show unusual artifacts or overly perfect patterns.",
            
            'saliency': f"This heatmap highlights WHERE our AI model focused when analyzing the audio. Brighter/warmer areas got more attention. The model classified this as {prediction} with {confidence:.1%} confidence. These focus areas help us understand what audio features triggered the classification.",
            
            'gradcam': f"This attention map shows which parts of the audio the AI considered most important. Red/yellow = high focus, blue = low focus. The model classified this as {prediction} ({confidence:.1%} confidence). This visualization helps verify the AI is looking at meaningful audio features, not making random guesses.",
            
            'pitch': f"This chart tracks how the voice's pitch (high/low tone) changes over time. Natural speech has subtle micro-variations in pitch, while synthetic voices may be too smooth or have unnatural jumps. This audio was classified as {prediction} with {confidence:.1%} confidence based partly on these pitch patterns.",
            
            'centroid': f"This graph shows the audio's 'brightness' over time - where the center of the sound's frequency content lies. Higher values mean brighter, sharper sounds. This audio was classified as {prediction} ({confidence:.1%} confidence). Unusual brightness patterns can indicate synthetic audio.",
            
            'zcr': f"Zero Crossing Rate shows how often the sound wave crosses silence. High values indicate noisy sounds (like 's' or 'f'), low values indicate vowel sounds. This audio was classified as {prediction} with {confidence:.1%} confidence. Unnatural patterns in voiced vs. unvoiced sounds can reveal deepfakes.",
            
            'rms': f"This shows the audio's volume/energy over time. Peaks are loud moments, valleys are quiet. This audio was classified as {prediction} ({confidence:.1%} confidence). Natural speech has dynamic range - synthetic audio may have unnaturally consistent or abrupt volume changes.",
            
            'shap': f"These bars show which audio features most influenced the AI's decision. Longer bars = more important features. Green/positive pushed toward one classification, red/negative toward the other. This helps explain WHY the AI classified this as {prediction} with {confidence:.1%} confidence.",
            
            'voice_analysis': f"This visualization shows detailed voice characteristics including pitch patterns and spectral features. The analysis classified this as {prediction} with {confidence:.1%} confidence based on voice authenticity indicators.",
            
            'segment_analysis': f"This shows how different segments of the audio were analyzed for consistency. Inconsistent patterns may indicate edited or spliced audio. The overall classification is {prediction} ({confidence:.1%} confidence)."
        }
        
        explanation = static_explanations.get(viz_type, f"This visualization helps analyze the audio, which was classified as {prediction} with {confidence:.1%} confidence. It provides visual evidence to support the AI's decision.")
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'viz_type': viz_type,
            'provider': 'fallback',
            'source': 'static'
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/api/resnet_v2/warmup', methods=['POST'])
def api_resnet_v2_warmup():
    """Preload the ResNet v2 backup model to avoid first-click latency during demos."""
    try:
        pipe = _get_hf_resnet_v2_pipeline()
        # Run a tiny dummy inference to force weights/tokenizer/feature extractor init.
        dummy_audio = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
        _ = pipe({"array": dummy_audio, "sampling_rate": SAMPLE_RATE})
        return jsonify({'success': True, 'status': 'warm'})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/api/references')
def api_references():
    """Get all research references"""
    return jsonify(RESEARCH_REFERENCES)


# ==========================================
# 🤖 GEMINI AI ENDPOINTS
# ==========================================

@app.route('/api/ai/status', methods=['GET'])
def api_ai_status():
    """Get the current AI service status."""
    global gemini_service
    
    if gemini_service is None and GEMINI_API_KEY:
        gemini_service = get_gemini_service(GEMINI_API_KEY)
    
    if gemini_service and gemini_service.is_available():
        return jsonify({
            'success': True,
            'available': True,
            'provider': 'gemini',
            'model': 'gemini-1.5-flash',
            'status': 'ready'
        })
    else:
        error = gemini_service.error if gemini_service else 'No API key configured'
        return jsonify({
            'success': True,
            'available': False,
            'provider': 'none',
            'status': 'not configured',
            'error': error,
            'hint': 'Set GEMINI_API_KEY environment variable or call /api/ai/configure'
        })


@app.route('/api/ai/configure', methods=['POST'])
def api_ai_configure():
    """Configure the AI service with an API key."""
    global gemini_service, GEMINI_API_KEY
    
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'success': False, 'error': 'No API key provided'})
        
        # Update the global key and reinitialize service
        GEMINI_API_KEY = api_key
        gemini_service = get_gemini_service(api_key)
        
        if gemini_service.is_available():
            return jsonify({
                'success': True,
                'message': 'Gemini AI configured successfully!',
                'provider': 'gemini',
                'model': 'gemini-1.5-flash'
            })
        else:
            return jsonify({
                'success': False,
                'error': gemini_service.error or 'Failed to initialize Gemini'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/llm/warmup', methods=['POST'])
def api_llm_warmup():
    """Warmup the AI service (Gemini doesn't need warmup, but kept for compatibility)."""
    global gemini_service
    
    # Initialize Gemini if not already done
    if gemini_service is None and GEMINI_API_KEY:
        gemini_service = get_gemini_service(GEMINI_API_KEY)
    
    if gemini_service and gemini_service.is_available():
        return jsonify({
            'success': True,
            'status': 'Gemini AI ready (no warmup needed)',
            'provider': 'gemini'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Gemini not configured. Set API key via /api/ai/configure',
            'hint': 'Get your API key from https://makersuite.google.com/app/apikey'
        })


@app.route('/api/explain', methods=['POST'])
def api_explain():
    """Generate a plain-English explanation of analysis results using Gemini AI."""
    global gemini_service
    
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        # Initialize Gemini if not already done
        if gemini_service is None and GEMINI_API_KEY:
            gemini_service = get_gemini_service(GEMINI_API_KEY)
        
        # Use Gemini service if available
        if gemini_service and gemini_service.is_available():
            result = gemini_service.explain_analysis_result(data)
            return jsonify(result)
        
        # Fallback to static explanation
        prediction = data.get('prediction', 'UNKNOWN')
        confidence = data.get('confidence', 0)
        
        if prediction == 'REAL':
            explanation = f"✅ Our analysis indicates this audio is likely GENUINE with {confidence:.1%} confidence. The voice patterns and audio characteristics appear natural and consistent with real human speech. You can generally trust this audio, but always consider the source and context."
        else:
            explanation = f"⚠️ Our analysis indicates this audio may be SYNTHETIC or manipulated ({confidence:.1%} confidence). Some characteristics suggest it might not be authentic human speech. We recommend exercising caution and verifying from other sources before trusting this audio."
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'provider': 'fallback'
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


# ==========================================
# 🚀 MAIN
# ==========================================

# Serve React app (catch-all route for SPA)
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    """Serve static files or fallback to index.html for SPA routing"""
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # Try to load default model on startup
    default_path = find_default_model()
    if default_path:
        print(f"🔄 Loading default model: {default_path}")
        success, result = load_model_file(default_path, "CNN")
        if success:
            print(f"✅ Model loaded: {result}")
        else:
            print(f"❌ Failed to load model: {result}")
    
    print("\n🌐 Starting Audio Deepfake Analyzer Web Server...")
    print("📍 Open http://localhost:5000 in your browser\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
