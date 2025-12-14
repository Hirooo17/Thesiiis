import React, { useEffect, useMemo, useRef, useState } from 'react';
import Lightbox from './components/Lightbox.jsx';
import Modal from './components/Modal.jsx';
import Tabs from './components/Tabs.jsx';
import { ToastHost } from './components/ToastHost.jsx';
import AnalyzeTab from './pages/AnalyzeTab.jsx';
import ExplainabilityTab from './pages/ExplainabilityTab.jsx';
import ReferencesTab from './pages/ReferencesTab.jsx';
import { apiGet, apiPostForm, apiPostJson } from './lib/api.js';
import { pickRecorderMimeType } from './lib/audio.js';
import { useToasts } from './hooks/useToasts.js';

const DURATION_OPTIONS = [3, 5, 7, 10, 15, 20, 30];
const SNR_LEVELS = [5, 10, 15, 20, 25, 30];

export default function App() {
  const [references, setReferences] = useState({});
  const { toasts, pushToast, dismissToast } = useToasts();

  // Model state
  const [modelType, setModelType] = useState('CNN');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState('No model loaded');

  // Audio state
  const [audioLoaded, setAudioLoaded] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');

  // Recording state
  const [recordingDuration, setRecordingDuration] = useState(3);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const recordingStopTimeoutRef = useRef(null);
  const recordingStartMsRef = useRef(0);
  const chunksRef = useRef([]);

  // Noise state
  const [snrValue, setSnrValue] = useState('None');
  const [customNoiseFile, setCustomNoiseFile] = useState(null);
  const [customNoiseUrl, setCustomNoiseUrl] = useState('');
  const [noiseMixLevel, setNoiseMixLevel] = useState(30);

  // UI state
  const [activeTab, setActiveTab] = useState('analyze');
  const [loading, setLoading] = useState({ open: false, title: 'Loading...', subtitle: 'Please wait' });
  const [infoModal, setInfoModal] = useState({ open: false, title: '', content: null });
  const [lightbox, setLightbox] = useState({ open: false, src: '', alt: '', vizType: null, result: null });

  // Result state
  const [result, setResult] = useState(null);
  const [llmExplanation, setLlmExplanation] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);

  // Explainability state
  const [gradcam, setGradcam] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [shap, setShap] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);

  const canAnalyze = modelLoaded && audioLoaded;
  const canAnalyzeResnetV2 = audioLoaded;
  const audioBlobOrFile = audioFile || audioBlob || null;

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (customNoiseUrl) URL.revokeObjectURL(customNoiseUrl);
    };
  }, []);

  // Load references + model status on mount
  useEffect(() => {
    (async () => {
      try {
        const refs = await apiGet('/api/references');
        setReferences(refs || {});
      } catch {
        // optional
      }

      try {
        const status = await apiGet('/api/model/status');
        if (status?.loaded) {
          setModelLoaded(true);
          setModelType(status.type || 'CNN');
          setModelName(status.name || 'Loaded');
        }
      } catch {
        // optional
      }
    })();
  }, []);

  // Info content for modals
  const infoContent = useMemo(() => {
    const listRefs = (key) => (references?.[key] || []).map((r) => ({ name: r.name, url: r.url }));

    return {
      waveform: {
        title: 'Waveform',
        bullets: ['Tall waves = loud sounds', 'Flat areas = silence', 'Regular patterns = consistent voice'],
        refs: listRefs('waveform')
      },
      spectrogram: {
        title: 'Mel Spectrogram',
        bullets: ['Bright = strong frequencies', 'Dark = quiet/no sound', 'AI fakes may show unnatural patterns'],
        refs: listRefs('spectrogram')
      },
      saliency: {
        title: 'Saliency Heatmap',
        bullets: ['Highlights where the model focused', 'Red = important regions', 'Used for explainability'],
        refs: listRefs('saliency')
      },
      mfcc: {
        title: 'Audio Features',
        bullets: ['Spectral centroid = brightness', 'Zero crossing rate = voiced/unvoiced behavior', 'RMS energy = loudness over time'],
        refs: listRefs('mfcc')
      },
      confidence: {
        title: 'Confidence Score',
        bullets: ['80%+ = very confident', '60–80% = moderate', '<60% = uncertain'],
        refs: listRefs('confidence')
      },
      voice_analysis: {
        title: 'Voice Analysis',
        bullets: ['Detects single vs multiple voices', 'Uses pitch (F0) and spectral variation', 'Helps flag mixed sources'],
        refs: listRefs('voice_analysis')
      },
      noise: {
        title: 'Noise & SNR',
        bullets: ['Higher SNR = cleaner audio', 'Noise can mask deepfake artifacts', 'Backend estimates SNR and impact'],
        refs: listRefs('noise')
      },
      mixed_audio: {
        title: 'Mixed Real/AI Detection',
        bullets: ['Segments audio and checks consistency', 'Warns when real+fake segments both present', 'Only available for CNN in backend'],
        refs: []
      }
    };
  }, [references]);

  const openInfo = (key) => {
    const info = infoContent[key] || { title: 'Info', bullets: [], refs: [] };
    setInfoModal({
      open: true,
      title: info.title,
      content: (
        <div className="space-y-3">
          {info.bullets?.length ? (
            <ul className="list-disc space-y-1 pl-5 text-zinc-200">
              {info.bullets.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          ) : null}
          {info.refs?.length ? (
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Research References</div>
              <ul className="space-y-1">
                {info.refs.map((r) => (
                  <li key={r.url}>
                    <a className="text-sm text-blue-300 hover:underline" href={r.url} target="_blank" rel="noreferrer">
                      {r.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      )
    });
  };

  const setLoadingOpen = (open, title, subtitle) => {
    setLoading((prev) => ({
      open,
      title: title ?? prev.title,
      subtitle: subtitle ?? prev.subtitle
    }));
  };

  // Audio file selection
  const onSelectAudioFile = (file) => {
    if (!file) return;
    setAudioFile(file);
    setAudioBlob(null);
    setAudioLoaded(true);
    setResult(null);
    setLlmExplanation(null);
    setGradcam(null);
    setShap(null);

    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    pushToast(`Audio loaded: ${file.name}`, 'success');
  };

  // Model file upload
  const onSelectModelFile = async (file) => {
    if (!file) return;

    setLoadingOpen(true, 'Uploading model...', 'Please wait');

    try {
      const form = new FormData();
      form.append('model', file);
      form.append('type', modelType);
      const data = await apiPostForm('/api/model/upload', form);

      setModelLoaded(true);
      setModelName(data.model_name || file.name);
      pushToast('Model uploaded and loaded', 'success');
    } catch (e) {
      pushToast(e.message || 'Failed to upload model', 'error');
    } finally {
      setLoadingOpen(false);
    }
  };

  // Load default model
  const loadDefaultModel = async () => {
    if (modelType !== 'CNN') {
      pushToast('Default model is only available for CNN (ResNet). Upload a Traditional ML model instead.', 'info');
      return;
    }

    setLoadingOpen(true, 'Loading model...', 'This may take a moment');

    try {
      const data = await apiPostJson('/api/model/load', { type: modelType });
      setModelLoaded(true);
      setModelName(data.model_name || 'Loaded');
      pushToast('Model loaded successfully', 'success');
    } catch (e) {
      setModelLoaded(false);
      setModelName('No model loaded');
      pushToast(e.message || 'Failed to load model', 'error');
    } finally {
      setLoadingOpen(false);
    }
  };

  // Recording functions
  const stopRecording = () => {
    try {
      if (recordingStopTimeoutRef.current) {
        clearTimeout(recordingStopTimeoutRef.current);
        recordingStopTimeoutRef.current = null;
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }

      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
      }
    } catch {
      // ignore
    } finally {
      setIsRecording(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = pickRecorderMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' });

        setAudioBlob(blob);
        setAudioFile(null);
        setAudioLoaded(true);
        setResult(null);
        setLlmExplanation(null);
        setGradcam(null);
        setShap(null);

        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(URL.createObjectURL(blob));

        stream.getTracks().forEach((t) => t.stop());

        const secs = Math.max(0, Math.round((performance.now() - recordingStartMsRef.current) / 1000));
        pushToast(`Recording ready (${secs}s)`, 'success');
      };

      mediaRecorderRef.current = recorder;
      recordingStartMsRef.current = performance.now();
      setRecordingSeconds(0);
      setIsRecording(true);

      recorder.start(250);

      recordingTimerRef.current = setInterval(() => {
        const elapsed = (performance.now() - recordingStartMsRef.current) / 1000;
        setRecordingSeconds(Math.min(recordingDuration, Math.floor(elapsed)));
      }, 200);

      recordingStopTimeoutRef.current = setTimeout(() => {
        stopRecording();
      }, recordingDuration * 1000 + 120);
    } catch (e) {
      pushToast(e?.message || 'Microphone permission denied', 'error');
      setIsRecording(false);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) stopRecording();
    else await startRecording();
  };

  // Custom noise handling
  const onPickCustomNoise = async (file) => {
    if (!file) return;
    setCustomNoiseFile(file);

    if (customNoiseUrl) URL.revokeObjectURL(customNoiseUrl);
    const url = URL.createObjectURL(file);
    setCustomNoiseUrl(url);

    setSnrValue('None');
    pushToast(`Noise file loaded: ${file.name}`, 'success');
  };

  const removeCustomNoise = () => {
    setCustomNoiseFile(null);
    if (customNoiseUrl) URL.revokeObjectURL(customNoiseUrl);
    setCustomNoiseUrl('');
    pushToast('Custom noise removed', 'info');
  };

  // Analysis functions
  const analyze = async () => {
    if (!canAnalyze) return;
    setLoadingOpen(true, 'Analyzing audio...', 'Please wait');
    setLlmExplanation(null);
    setGradcam(null);
    setShap(null);

    try {
      const form = new FormData();
      form.append('type', modelType);

      if (audioFile) {
        form.append('audio', audioFile);
      } else if (audioBlob) {
        const ext = audioBlob.type?.includes('ogg') ? 'ogg' : 'webm';
        form.append('audio', audioBlob, `recording.${ext}`);
      }

      form.append('snr', snrValue);

      if (customNoiseFile) {
        form.append('custom_noise', customNoiseFile);
        form.append('noise_mix_level', String(noiseMixLevel));
      }

      const data = await apiPostForm('/api/analyze', form);
      setResult(data);
      pushToast('Analysis complete', 'success');
    } catch (e) {
      pushToast(e.message || 'Analysis failed', 'error');
    } finally {
      setLoadingOpen(false);
    }
  };

  const warmupResnetV2 = async () => {
    setLoadingOpen(true, 'Warming up ResNet v2...', 'One-time load can take a minute');
    try {
      await apiPostJson('/api/resnet_v2/warmup', {});
      pushToast('ResNet v2 warmed up', 'success');
    } catch (e) {
      pushToast(e.message || 'Warm-up failed', 'error');
    } finally {
      setLoadingOpen(false);
    }
  };

  const analyzeResnetV2 = async () => {
    if (!canAnalyzeResnetV2) return;

    setLoadingOpen(true, 'Analyzing audio...', 'Using ResNet v2');
    setLlmExplanation(null);
    setGradcam(null);
    setShap(null);

    try {
      const form = new FormData();
      if (audioFile) {
        form.append('audio', audioFile);
      } else if (audioBlob) {
        const ext = audioBlob.type?.includes('ogg') ? 'ogg' : 'webm';
        form.append('audio', audioBlob, `recording.${ext}`);
      }

      form.append('snr', snrValue);

      if (customNoiseFile) {
        form.append('custom_noise', customNoiseFile);
        form.append('noise_mix_level', String(noiseMixLevel));
      }

      const data = await apiPostForm('/api/analyze_resnet_v2', form);
      setResult(data);
      pushToast('Analysis complete (ResNet v2)', 'success');
    } catch (e) {
      pushToast(e.message || 'ResNet v2 analysis failed', 'error');
    } finally {
      setLoadingOpen(false);
    }
  };

  // LLM explanation functions
  const warmupLlm = async () => {
    setLlmLoading(true);
    pushToast('Warming up AI model (may take 1–2 min first time)...', 'info');
    try {
      const data = await apiPostJson('/api/llm/warmup', {});
      if (data.success) pushToast('AI model ready!', 'success');
      else pushToast(data.error || 'LLM warmup failed', 'error');
    } catch (e) {
      pushToast(e.message || 'Could not warm up AI', 'error');
    } finally {
      setLlmLoading(false);
    }
  };

  const explainResults = async () => {
    if (!result) return;
    setLlmLoading(true);
    setLlmExplanation(null);
    pushToast('Generating explanation (may take 30–60s)...', 'info');

    try {
      const data = await apiPostJson('/api/explain', {
        prediction: result.prediction,
        confidence: result.confidence,
        adjusted_confidence: result.adjusted_confidence,
        voice_analysis: result.voice_analysis,
        noise_analysis: result.noise_analysis,
        mixed_audio_warning: result.mixed_audio_warning,
        audio_duration: result.audio_duration
      });

      if (data.success) {
        setLlmExplanation(data.explanation);
        pushToast('AI explanation generated', 'success');
      } else {
        pushToast(data.error || 'Failed to generate explanation', 'error');
      }
    } catch (e) {
      pushToast(e.message || 'Could not connect to AI service', 'error');
    } finally {
      setLlmLoading(false);
    }
  };

  // Explainability functions
  const requestGradcam = async () => {
    if (!audioLoaded) return;
    setGradcamLoading(true);
    pushToast('Generating Grad-CAM…', 'info');
    try {
      const form = new FormData();
      if (audioFile) form.append('audio', audioFile);
      else if (audioBlob) {
        const ext = audioBlob.type?.includes('ogg') ? 'ogg' : 'webm';
        form.append('audio', audioBlob, `recording.${ext}`);
      }
      form.append('snr', snrValue);
      if (customNoiseFile) {
        form.append('custom_noise', customNoiseFile);
        form.append('noise_mix_level', String(noiseMixLevel));
      }

      const data = await apiPostForm('/api/gradcam', form);
      if (data?.success && data.gradcam) {
        setGradcam(data.gradcam);
        pushToast('Grad-CAM ready', 'success');
      } else {
        pushToast(data?.error || 'Grad-CAM unavailable', 'error');
      }
    } catch (e) {
      pushToast(e.message || 'Failed to generate Grad-CAM', 'error');
    } finally {
      setGradcamLoading(false);
    }
  };

  const requestShap = async () => {
    if (!audioLoaded) return;
    setShapLoading(true);
    pushToast('Computing SHAP…', 'info');
    try {
      const form = new FormData();
      if (audioFile) form.append('audio', audioFile);
      else if (audioBlob) {
        const ext = audioBlob.type?.includes('ogg') ? 'ogg' : 'webm';
        form.append('audio', audioBlob, `recording.${ext}`);
      }
      const data = await apiPostForm('/api/shap', form);
      setShap(data);
      if (data?.available) pushToast('SHAP ready', 'success');
      else pushToast(data?.error || 'SHAP unavailable', 'error');
    } catch (e) {
      pushToast(e.message || 'Failed to compute SHAP', 'error');
    } finally {
      setShapLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <div className="sticky top-0 z-40 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur animate-fade-in">
        <div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div className="min-w-0 animate-slide-in-left">
            <div className="text-xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient">
                ✨ Veritas.AI
              </span>
            </div>
            <div className="text-xs text-zinc-400">Deepfake Audio Detection • Thesis Defense Edition</div>
          </div>

          <Tabs
            tabs={[
              { key: 'analyze', label: '🎤 Analyze' },
              { key: 'explain', label: '🔍 Explainability' },
              { key: 'references', label: '📚 References' }
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          <div className={`flex items-center gap-2 rounded-xl border px-3 py-2 animate-slide-in-right transition-all duration-300 ${
            modelLoaded 
              ? 'border-emerald-500/30 bg-emerald-950/30' 
              : 'border-zinc-800 bg-zinc-900'
          }`}>
            <div className={`h-2.5 w-2.5 rounded-full transition-all duration-300 ${
              modelLoaded ? 'bg-emerald-400 animate-success-pulse' : 'bg-rose-400 animate-danger-pulse'
            }`} />
            <div className="text-sm text-zinc-200">
              {modelLoaded ? (
                <span>
                  <span className="text-emerald-300">✓ Loaded:</span> {modelName}
                </span>
              ) : (
                <span className="text-zinc-300">⚠ No model loaded</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Tab Content with animation */}
      <div key={activeTab} className="animate-fade-in">
        {activeTab === 'analyze' && (
          <AnalyzeTab
            references={references}
            modelType={modelType}
            setModelType={setModelType}
            modelLoaded={modelLoaded}
            modelName={modelName}
            onLoadDefaultModel={loadDefaultModel}
          onUploadModel={onSelectModelFile}
          recordingDuration={recordingDuration}
          setRecordingDuration={setRecordingDuration}
          durationOptions={DURATION_OPTIONS}
          isRecording={isRecording}
          recordingSeconds={recordingSeconds}
          onToggleRecording={toggleRecording}
          audioLoaded={audioLoaded}
          audioUrl={audioUrl}
          onImportAudio={onSelectAudioFile}
          snrValue={snrValue}
          setSnrValue={setSnrValue}
          snrLevels={SNR_LEVELS}
          customNoiseFile={customNoiseFile}
          customNoiseUrl={customNoiseUrl}
          onPickCustomNoise={onPickCustomNoise}
          onRemoveCustomNoise={removeCustomNoise}
          noiseMixLevel={noiseMixLevel}
          setNoiseMixLevel={setNoiseMixLevel}
          audioBlobOrFile={audioBlobOrFile}
          pushToast={pushToast}
          canAnalyze={canAnalyze}
          onAnalyze={analyze}
          canAnalyzeResnetV2={canAnalyzeResnetV2}
          onAnalyzeResnetV2={analyzeResnetV2}
          onWarmupResnetV2={warmupResnetV2}
          result={result}
          llmLoading={llmLoading}
          onExplain={explainResults}
          onWarmupLlm={warmupLlm}
          llmExplanation={llmExplanation}
          onOpenInfo={openInfo}
        />
        )}

        {activeTab === 'explain' && (
          <ExplainabilityTab
            result={result}
            onZoomImage={(src, alt, vizType) => setLightbox({ open: true, src, alt: alt || 'Visualization', vizType, result })}
            onRequestGradcam={requestGradcam}
            gradcam={gradcam}
            gradcamLoading={gradcamLoading}
            shap={shap}
            shapLoading={shapLoading}
            onRequestShap={requestShap}
          />
        )}

        {activeTab === 'references' && <ReferencesTab references={references} />}
      </div>

      {/* Loading Overlay */}
      {loading.open && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4">
          <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-950 p-6 shadow-xl">
            <div className="flex items-center gap-3">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-zinc-100" />
              <div>
                <div className="text-sm font-semibold text-zinc-100">{loading.title}</div>
                <div className="text-xs text-zinc-400">{loading.subtitle}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modal */}
      <Modal open={infoModal.open} title={infoModal.title} onClose={() => setInfoModal({ open: false, title: '', content: null })}>
        {infoModal.content}
      </Modal>

      {/* Lightbox */}
      <Lightbox
        open={lightbox.open}
        src={lightbox.src}
        alt={lightbox.alt}
        vizType={lightbox.vizType}
        result={lightbox.result}
        onClose={() => setLightbox({ open: false, src: '', alt: '', vizType: null, result: null })}
      />

      {/* Toasts */}
      <ToastHost toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
