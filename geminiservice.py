"""
Gemini AI Service for Veritas.AI
================================
Provides AI-powered explanations for audio deepfake detection visualizations
using Google's Gemini API.

Usage:
    from geminiservice import GeminiService
    
    service = GeminiService(api_key="your-api-key")
    explanation = service.explain_visualization("waveform", "FAKE", 0.85, context)
"""

import os
import requests
from typing import Optional, Dict, Any

# We use the REST API directly to avoid protobuf conflicts with TensorFlow
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiService:
    """
    Gemini AI service for generating human-friendly explanations.
    Uses the REST API directly to avoid protobuf conflicts with TensorFlow.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini service.
        
        Args:
            api_key: Google AI API key. If not provided, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.model_name = 'gemini-1.5-flash'
        self.initialized = False
        self.error = None
            
        if not self.api_key:
            self.error = "No API key provided"
            return
        
        # Test the API key with a simple request
        try:
            test_url = f"{GEMINI_API_BASE}/models/{self.model_name}?key={self.api_key}"
            resp = requests.get(test_url, timeout=10)
            if resp.status_code == 200:
                self.initialized = True
                print("✅ Gemini AI service initialized successfully")
            else:
                self.error = f"API key validation failed: {resp.status_code}"
                print(f"❌ Gemini API error: {resp.text}")
        except Exception as e:
            self.error = str(e)
            print(f"❌ Failed to initialize Gemini: {e}")
    
    def is_available(self) -> bool:
        """Check if the service is available."""
        return self.initialized and bool(self.api_key)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'available': self.is_available(),
            'provider': 'gemini',
            'model': self.model_name,
            'error': self.error
        }
    
    def _call_gemini(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Call Gemini API via REST."""
        if not self.is_available():
            return None
        
        url = f"{GEMINI_API_BASE}/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        return parts[0].get('text', '').strip()
            else:
                print(f"Gemini API error: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Gemini request error: {e}")
        
        return None
    
    def _get_visualization_prompt(self, viz_type: str, prediction: str, confidence: float, context: Dict[str, Any]) -> str:
        """Generate the appropriate prompt for each visualization type."""
        
        audio_duration = context.get('audio_duration', 0)
        pitch_std = context.get('pitch_std', 0)
        spectral_variation = context.get('spectral_variation', 0)
        snr = context.get('snr', 0)
        
        prompts = {
            'waveform': f"""You are explaining an audio WAVEFORM visualization to someone with no technical background.

Context:
- The audio was analyzed by an AI deepfake detector
- Classification result: {prediction} with {confidence:.1%} confidence
- Audio duration: {audio_duration:.2f} seconds

What is a waveform: A waveform shows sound pressure (loudness) over time. The horizontal axis is time, vertical is amplitude (volume). Peaks = loud, valleys = quiet.

Your task: Write a friendly, simple 3-4 sentence explanation that:
1. Describes what they're seeing (wavy lines = sound)
2. Explains what patterns suggest REAL vs FAKE audio
3. Relates it to the {prediction} result
4. Uses everyday language, no jargon""",

            'spectrogram': f"""You are explaining a MEL SPECTROGRAM visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

What is a spectrogram: A colorful "heat map" of sound showing:
- Horizontal: Time (left to right)
- Vertical: Frequency/pitch (low at bottom, high at top)
- Colors: Intensity (brighter = louder)

Your task: Write a friendly 3-4 sentence explanation that:
1. Describes what the colorful image represents
2. Explains patterns that indicate REAL vs FAKE
3. Why this helps detect deepfakes
4. Uses everyday language, no jargon""",

            'saliency': f"""You are explaining a SALIENCY HEATMAP to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

What is a saliency map: It shows WHERE the AI "looked" when deciding. Brighter/warmer colors = more attention.

Your task: Write a friendly 3-4 sentence explanation that:
1. Explains what the highlighted regions mean
2. Why certain audio parts matter more for detection
3. How this builds trust in the AI's decision
4. Uses everyday language, no jargon""",

            'gradcam': f"""You are explaining a GRADIENT ATTENTION MAP to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

What is this: Shows which audio parts the AI focused on. Red/yellow = high attention, blue = low attention.

Your task: Write a friendly 3-4 sentence explanation that:
1. What the colored regions indicate
2. Why the AI focused on specific parts
3. How this verifies the AI isn't guessing randomly
4. Uses everyday language, no jargon""",

            'pitch': f"""You are explaining a PITCH CONTOUR visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence
- Pitch variation: {pitch_std:.2f} Hz

What is pitch contour: Shows how voice pitch (high/low tone) changes over time. Like a melody line of speech.

Your task: Write a friendly 3-4 sentence explanation that:
1. What the up-and-down line represents
2. What natural vs synthetic pitch looks like
3. What {pitch_std:.2f} Hz variation suggests
4. Uses everyday language, no jargon""",

            'centroid': f"""You are explaining a SPECTRAL CENTROID visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence
- Spectral variation: {spectral_variation:.4f}

What is spectral centroid: Measures audio "brightness" - higher = sharper, brighter sound; lower = duller, warmer.

Your task: Write a friendly 3-4 sentence explanation that:
1. What the fluctuating line represents
2. Why brightness patterns matter for detection
3. What the variation value tells us
4. Uses everyday language, no jargon""",

            'zcr': f"""You are explaining a ZERO CROSSING RATE (ZCR) visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

What is ZCR: Counts how often the sound wave crosses silence. High = noisy sounds like "s", low = vowel sounds like "a".

Your task: Write a friendly 3-4 sentence explanation that:
1. What the spiky pattern represents
2. Why this helps detect deepfakes
3. What unusual patterns might indicate
4. Uses everyday language, no jargon""",

            'rms': f"""You are explaining an RMS ENERGY (loudness) visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

What is RMS Energy: Measures volume over time. Peaks = loud moments, valleys = quiet moments.

Your task: Write a friendly 3-4 sentence explanation that:
1. What peaks and valleys represent
2. Why loudness patterns matter for detection
3. What this pattern suggests about the audio
4. Uses everyday language, no jargon""",

            'shap': f"""You are explaining SHAP FEATURE IMPORTANCE values to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

What is SHAP: Shows which audio characteristics most influenced the AI's decision. Positive = pushes toward one result, negative = toward the other.

Your task: Write a friendly 3-4 sentence explanation that:
1. What the feature bars represent
2. How to read positive/negative values
3. Why this helps us trust the AI
4. Uses everyday language, no jargon""",

            'voice_analysis': f"""You are explaining a VOICE ANALYSIS visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence
- Pitch variation: {pitch_std:.2f} Hz

This shows detailed analysis of voice characteristics including pitch patterns and spectral features.

Your task: Write a friendly 3-4 sentence explanation that:
1. What voice analysis reveals about the audio
2. Key indicators of natural vs synthetic voice
3. How this supports the {prediction} classification
4. Uses everyday language, no jargon""",

            'segment_analysis': f"""You are explaining a SEGMENT ANALYSIS visualization to someone with no technical background.

Context:
- Classification result: {prediction} with {confidence:.1%} confidence

This shows how different segments of the audio were analyzed for consistency.

Your task: Write a friendly 3-4 sentence explanation that:
1. What segment-by-segment analysis reveals
2. Why consistency matters for detection
3. What patterns suggest mixed or spliced audio
4. Uses everyday language, no jargon"""
        }
        
        return prompts.get(viz_type, f"""You are explaining an audio analysis visualization to someone with no technical background.

The audio was classified as {prediction} with {confidence:.1%} confidence.
Visualization type: {viz_type}

Write a simple, friendly 3-4 sentence explanation of what this shows and why it matters for detecting deepfake audio.""")
    
    def explain_visualization(self, viz_type: str, prediction: str, confidence: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an explanation for a specific visualization.
        
        Args:
            viz_type: Type of visualization (waveform, spectrogram, saliency, etc.)
            prediction: Model prediction (REAL or FAKE)
            confidence: Confidence score (0-1)
            context: Additional context (audio_duration, pitch_std, etc.)
        
        Returns:
            Dict with 'success', 'explanation', and other metadata
        """
        if not self.is_available():
            return self._get_fallback_explanation(viz_type, prediction, confidence)
        
        context = context or {}
        prompt = self._get_visualization_prompt(viz_type, prediction, confidence, context)
        
        explanation = self._call_gemini(prompt, max_tokens=300)
        
        if explanation:
            return {
                'success': True,
                'explanation': explanation,
                'viz_type': viz_type,
                'provider': 'gemini',
                'model': self.model_name
            }
        else:
            # Fall back to static explanation
            return self._get_fallback_explanation(viz_type, prediction, confidence)
    
    def explain_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation of the full analysis result.
        
        Args:
            result: The complete analysis result dictionary
        
        Returns:
            Dict with 'success', 'explanation', and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': self.error or 'Gemini service not available',
                'explanation': self._get_simple_fallback_explanation(result)
            }
        
        prediction = result.get('prediction', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        adjusted_confidence = result.get('adjusted_confidence', confidence)
        audio_duration = result.get('audio_duration', 0)
        voice_analysis = result.get('voice_analysis', {})
        noise_analysis = result.get('noise_analysis', {})
        mixed_audio_warning = result.get('mixed_audio_warning', '')
        
        voice_type = voice_analysis.get('source_type', 'unknown')
        voice_explanation = voice_analysis.get('explanation', '')
        noise_percent = noise_analysis.get('noise_percent', 0)
        snr = noise_analysis.get('snr_estimated', 0)
        noise_impact = noise_analysis.get('impact_description', '')
        
        prompt = f"""You are Veritas.AI, a friendly expert assistant explaining audio deepfake detection results to someone with no technical background.

ANALYSIS RESULTS:
- Verdict: {prediction}
- Confidence: {confidence:.1%} (Adjusted: {adjusted_confidence:.1%})
- Audio Duration: {audio_duration:.1f} seconds
- Voice Type: {voice_type}
- Noise Level: {noise_percent:.0f}% (SNR: {snr:.1f} dB)
- Noise Impact: {noise_impact}
{f'- ⚠️ Warning: {mixed_audio_warning}' if mixed_audio_warning else ''}

YOUR TASK:
Write a clear, friendly explanation in 4-5 sentences that:
1. States the verdict simply (is it likely real or fake?)
2. Explains what the confidence score means in plain terms
3. Mentions any relevant voice or audio quality observations
4. If there are warnings, explain what they mean
5. End with practical advice (e.g., "This appears trustworthy" or "Exercise caution with this audio")

Use a warm, helpful tone. Avoid jargon. Imagine explaining to a friend or family member."""

        explanation = self._call_gemini(prompt, max_tokens=400)
        
        if explanation:
            return {
                'success': True,
                'explanation': explanation,
                'provider': 'gemini',
                'model': self.model_name
            }
        else:
            return {
                'success': False,
                'error': 'Gemini API returned no response',
                'explanation': self._get_simple_fallback_explanation(result)
            }
    
    def _get_fallback_explanation(self, viz_type: str, prediction: str, confidence: float) -> Dict[str, Any]:
        """Get a static fallback explanation when Gemini is not available."""
        
        static = {
            'waveform': f"This waveform shows how the sound's loudness changes over time. The wavy pattern represents the audio signal - peaks are loud moments, valleys are quiet moments. This audio was classified as {prediction} with {confidence:.1%} confidence. Natural human speech typically shows varied, organic patterns, while synthetic audio may appear unnaturally smooth or repetitive.",
            
            'spectrogram': f"This colorful 'heat map' shows sound frequencies over time. Brighter colors indicate louder sounds at those frequencies. The horizontal axis is time, vertical is pitch (low to high). This audio was classified as {prediction} ({confidence:.1%} confidence). Real voices create natural harmonic patterns, while AI-generated audio may show unusual artifacts.",
            
            'saliency': f"This heatmap highlights WHERE our AI model focused when analyzing the audio. Brighter areas got more attention. The model classified this as {prediction} with {confidence:.1%} confidence. These focus areas help us understand what audio features triggered the classification.",
            
            'gradcam': f"This attention map shows which parts of the audio the AI considered most important. Red/yellow = high focus, blue = low focus. The model classified this as {prediction} ({confidence:.1%} confidence). This helps verify the AI is looking at meaningful features.",
            
            'pitch': f"This chart tracks how the voice's pitch changes over time. Natural speech has subtle micro-variations, while synthetic voices may be too smooth or have unnatural jumps. This audio was classified as {prediction} with {confidence:.1%} confidence.",
            
            'centroid': f"This graph shows the audio's 'brightness' over time. Higher values mean sharper sounds. This audio was classified as {prediction} ({confidence:.1%} confidence). Unusual brightness patterns can indicate synthetic audio.",
            
            'zcr': f"Zero Crossing Rate shows how often the sound wave crosses silence. High values = noisy sounds, low values = vowels. This audio was classified as {prediction} with {confidence:.1%} confidence. Unnatural patterns can reveal deepfakes.",
            
            'rms': f"This shows the audio's volume over time. Peaks are loud, valleys are quiet. This audio was classified as {prediction} ({confidence:.1%} confidence). Natural speech has dynamic range - synthetic may be unnaturally consistent.",
            
            'shap': f"These bars show which audio features most influenced the AI's decision. Longer bars = more important features. This helps explain WHY the AI classified this as {prediction} with {confidence:.1%} confidence.",
            
            'voice_analysis': f"This visualization shows detailed voice characteristics including pitch patterns and spectral features. The analysis supports the {prediction} classification with {confidence:.1%} confidence.",
            
            'segment_analysis': f"This shows how different segments of the audio were analyzed for consistency. Inconsistent segments may indicate edited or spliced audio. Classification: {prediction} ({confidence:.1%} confidence)."
        }
        
        explanation = static.get(viz_type, f"This visualization helps analyze the audio, which was classified as {prediction} with {confidence:.1%} confidence.")
        
        return {
            'success': True,
            'explanation': explanation,
            'viz_type': viz_type,
            'provider': 'fallback',
            'source': 'static'
        }
    
    def _get_simple_fallback_explanation(self, result: Dict[str, Any]) -> str:
        """Generate a simple fallback explanation for the full result."""
        prediction = result.get('prediction', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        
        if prediction == 'REAL':
            return f"✅ Our analysis indicates this audio is likely GENUINE with {confidence:.1%} confidence. The voice patterns and audio characteristics appear natural and consistent with real human speech. You can generally trust this audio, but always consider the source and context."
        else:
            return f"⚠️ Our analysis indicates this audio may be SYNTHETIC or manipulated ({confidence:.1%} confidence). Some characteristics suggest it might not be authentic human speech. We recommend exercising caution and verifying from other sources before trusting this audio."


# Singleton instance
_gemini_service: Optional[GeminiService] = None


def get_gemini_service(api_key: Optional[str] = None) -> GeminiService:
    """Get or create the Gemini service singleton."""
    global _gemini_service
    
    if _gemini_service is None or (api_key and api_key != _gemini_service.api_key):
        _gemini_service = GeminiService(api_key=api_key)
    
    return _gemini_service


def init_gemini(api_key: str) -> GeminiService:
    """Initialize the Gemini service with an API key."""
    return get_gemini_service(api_key)
