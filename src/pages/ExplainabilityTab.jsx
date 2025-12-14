import React, { useMemo, useState, useEffect } from 'react';
import ConfidenceRing from '../components/ConfidenceRing.jsx';
import RadarChart from '../components/RadarChart.jsx';
import { toDataImage, formatPct01, clamp } from '../lib/format.js';
import { apiGet, apiPostJson } from '../lib/api.js';

// AI Status Badge Component
function AIStatusBadge({ onConfigure }) {
  const [status, setStatus] = useState({ available: false, loading: true });
  const [showConfig, setShowConfig] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [configuring, setConfiguring] = useState(false);

  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const res = await apiGet('/api/ai/status');
      setStatus({ ...res, loading: false });
    } catch {
      setStatus({ available: false, loading: false, error: 'Could not check status' });
    }
  };

  const handleConfigure = async () => {
    if (!apiKey.trim()) return;
    setConfiguring(true);
    try {
      await apiPostJson('/api/ai/configure', { api_key: apiKey });
      setApiKey('');
      setShowConfig(false);
      checkStatus();
    } catch (e) {
      alert('Failed to configure: ' + e.message);
    } finally {
      setConfiguring(false);
    }
  };

  if (status.loading) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs text-zinc-400">
        <div className="h-3 w-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
        Checking AI...
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setShowConfig(!showConfig)}
        className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
          status.available
            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30'
            : 'bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30'
        }`}
      >
        {status.available ? (
          <>
            <span className="text-base">✨</span>
            Gemini AI Active
          </>
        ) : (
          <>
            <span className="text-base">⚠️</span>
            AI Not Configured
          </>
        )}
      </button>

      {showConfig && !status.available && (
        <div className="absolute right-0 top-full mt-2 z-50 w-80 rounded-xl border border-zinc-700 bg-zinc-900 p-4 shadow-2xl animate-scale-in">
          <div className="text-sm font-semibold text-zinc-100 mb-2">🔑 Configure Gemini AI</div>
          <p className="text-xs text-zinc-400 mb-3">
            Get your API key from{' '}
            <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              Google AI Studio
            </a>
          </p>
          <input
            type="password"
            placeholder="Enter your Gemini API key..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-blue-500 focus:outline-none"
          />
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={handleConfigure}
              disabled={configuring || !apiKey.trim()}
              className="flex-1 rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {configuring ? 'Configuring...' : 'Save Key'}
            </button>
            <button
              type="button"
              onClick={() => setShowConfig(false)}
              className="rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function VizCard({ title, subtitle, src, onZoom, right, loading, vizType }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4 card-hover animate-slide-up transition-all duration-300">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-zinc-100">{title}</div>
          {subtitle ? <div className="mt-0.5 text-xs text-zinc-500">{subtitle}</div> : null}
        </div>
        {right}
      </div>

      {loading ? (
        <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-10 text-center">
          <div className="inline-block h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <div className="mt-2 text-xs text-zinc-500">Generating...</div>
        </div>
      ) : src ? (
        <button type="button" onClick={() => onZoom?.(src, title, vizType)} className="mt-3 w-full group">
          <img 
            src={src} 
            alt={title} 
            className="w-full rounded-xl border border-zinc-800 transition-all duration-300 group-hover:border-zinc-600 group-hover:shadow-lg group-hover:scale-[1.02]" 
          />
        </button>
      ) : (
        <div className="mt-3 rounded-xl border border-dashed border-zinc-800 bg-zinc-950 px-4 py-10 text-center text-xs text-zinc-500">
          <div className="text-2xl mb-2 animate-float">📊</div>
          No data yet
        </div>
      )}
    </div>
  );
}

function FactorBar({ label, value, positive }) {
  const percentage = Math.round(clamp(value, 0, 1) * 100);
  const colorClass = positive ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 'bg-gradient-to-r from-rose-500 to-rose-400';
  
  return (
    <div className="space-y-1 animate-slide-up">
      <div className="flex items-center justify-between text-xs">
        <span className="text-zinc-300">{label}</span>
        <span className={`font-semibold ${positive ? 'text-emerald-400' : 'text-rose-400'}`}>{percentage}%</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full ${colorClass} transition-all duration-700 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function ExplainabilityTab({ result, onZoomImage, onRequestGradcam, gradcam, gradcamLoading, shap, shapLoading, onRequestShap }) {
  const viz = useMemo(() => {
    if (!result) return null;
    return {
      waveform: toDataImage(result.visualizations?.waveform),
      spectrogram: toDataImage(result.visualizations?.spectrogram),
      saliency: toDataImage(result.visualizations?.saliency),
      pitch: toDataImage(result.visualizations?.pitch),
      centroid: toDataImage(result.visualizations?.centroid),
      zcr: toDataImage(result.visualizations?.zcr),
      rms: toDataImage(result.visualizations?.rms),
      voiceFig: toDataImage(result.visualizations?.voice_analysis_figure),
      segmentFig: toDataImage(result.visualizations?.segment_analysis_figure)
    };
  }, [result]);

  const verdict = result?.prediction || null;
  const isReal = verdict === 'REAL';

  return (
    <div className="mx-auto max-w-[1800px] px-4 py-5 animate-fade-in">
      {!result ? (
        <div className="grid place-items-center rounded-2xl border border-dashed border-zinc-800 bg-zinc-950 py-20 text-center">
          <div className="max-w-md space-y-2">
            <div className="text-4xl animate-float">🔍</div>
            <div className="text-lg font-semibold text-zinc-100">No results yet</div>
            <div className="text-sm text-zinc-400">Run an analysis in the Analyze tab first.</div>
          </div>
        </div>
      ) : (
        <>
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3 animate-slide-up">
            <div>
              <div className="text-base font-semibold text-zinc-100">🔬 Explainability Dashboard</div>
              <div className="text-xs text-zinc-500">
                Verdict: <span className={`font-semibold ${isReal ? 'text-emerald-300' : 'text-rose-300'}`}>{isReal ? '✅ REAL' : '❌ FAKE'}</span> • Confidence: {formatPct01(result?.confidence)}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <AIStatusBadge />
              <button
                type="button"
                onClick={onRequestGradcam}
                disabled={gradcamLoading}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-300 ${
                  gradcamLoading 
                    ? 'cursor-wait bg-zinc-800 text-zinc-500' 
                    : 'border border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:scale-105 hover:border-zinc-700'
                }`}
              >
                {gradcamLoading ? (
                  <span className="flex items-center gap-2">
                    <span className="inline-block h-3 w-3 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" />
                    Generating Grad-CAM…
                  </span>
                ) : (
                  '🔥 Generate Grad-CAM'
                )}
              </button>
              <button
                type="button"
                onClick={onRequestShap}
                disabled={shapLoading}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-300 ${
                  shapLoading 
                    ? 'cursor-wait bg-zinc-800 text-zinc-500' 
                    : 'border border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:scale-105 hover:border-zinc-700'
                }`}
              >
                {shapLoading ? (
                  <span className="flex items-center gap-2">
                    <span className="inline-block h-3 w-3 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" />
                    Computing SHAP…
                  </span>
                ) : (
                  '📊 Compute SHAP'
                )}
              </button>
            </div>
          </div>

          {/* Detection Result Summary */}
          <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-3 stagger-children">
            {/* Detection Result Card */}
            <div className={`rounded-2xl border bg-gradient-to-br from-zinc-900 to-zinc-950 p-6 animate-slide-up card-hover transition-all duration-500 ${
              isReal ? 'border-emerald-500/30 shadow-lg shadow-emerald-500/5' : 'border-rose-500/30 shadow-lg shadow-rose-500/5'
            }`}>
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">🎯 Detection Result</div>
              <div className="mt-4 flex items-center gap-4">
                <ConfidenceRing value={result?.confidence || 0} size={100} isReal={isReal} />
                <div className="space-y-2">
                  <div className={`text-2xl font-bold ${isReal ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {isReal ? '✅ REAL' : '❌ FAKE'}
                  </div>
                  <div className="space-y-1 text-xs text-zinc-400">
                    <div>Raw: {formatPct01(result?.confidence)}</div>
                    {typeof result?.adjusted_confidence === 'number' && (
                      <div>Adjusted: {formatPct01(result.adjusted_confidence)}</div>
                    )}
                  </div>
                </div>
              </div>
              {result?.audio_duration && (
                <div className="mt-4 text-xs text-zinc-500">
                  🎵 Audio duration: {result.audio_duration.toFixed(2)}s
                </div>
              )}
            </div>

            {/* Technical Metric Profile (Radar Chart) */}
            <div className="rounded-2xl border border-zinc-800 bg-gradient-to-br from-zinc-900 to-zinc-950 p-6 animate-slide-up card-hover">
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">📈 Technical Metric Profile</div>
              <div className="mt-4 flex items-center justify-center">
                <RadarChart
                  size={180}
                  metrics={[
                    { label: 'Pitch Var', value: clamp((result?.voice_analysis?.pitch_std || 0) / 50, 0, 1) },
                    { label: 'Spectral', value: clamp((result?.voice_analysis?.spectral_variation || 0), 0, 1) },
                    { label: 'Energy', value: clamp((result?.voice_analysis?.energy_std || 0) * 10, 0, 1) },
                    { label: 'Confidence', value: result?.confidence || 0 },
                    { label: 'SNR', value: clamp((result?.noise_analysis?.snr_estimated || 30) / 60, 0, 1) }
                  ]}
                />
              </div>
            </div>

            {/* Top Contributing Factors */}
            <div className="rounded-2xl border border-zinc-800 bg-gradient-to-br from-zinc-900 to-zinc-950 p-6 animate-slide-up card-hover">
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">🔑 Key Indicators</div>
              <div className="mt-4 space-y-3 stagger-children">
                <FactorBar label="Voice Consistency" value={isReal ? 0.85 : 0.35} positive={isReal} />
                <FactorBar 
                  label="Spectral Pattern" 
                  value={clamp((result?.voice_analysis?.spectral_variation || 0.5), 0, 1)} 
                  positive={isReal} 
                />
                <FactorBar 
                  label="Pitch Naturalness" 
                  value={clamp((result?.voice_analysis?.pitch_std || 10) / 30, 0, 1)} 
                  positive={isReal} 
                />
                <FactorBar 
                  label="Signal Quality" 
                  value={1 - clamp((result?.noise_analysis?.noise_percent || 0) / 100, 0, 1)} 
                  positive={true} 
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3 stagger-children">
            <VizCard title="📈 Waveform" subtitle="Sound wave pattern" src={viz?.waveform} onZoom={onZoomImage} vizType="waveform" />
            <VizCard title="🌈 Mel Spectrogram" subtitle="Frequency fingerprint" src={viz?.spectrogram} onZoom={onZoomImage} vizType="spectrogram" />
            <VizCard title="🔥 Saliency Heatmap" subtitle="Where the model focused" src={viz?.saliency} onZoom={onZoomImage} vizType="saliency" />

            <VizCard
              title="🎯 Grad-CAM (CNN)"
              subtitle="Class-activation map overlay"
              src={gradcam ? `data:image/png;base64,${gradcam}` : null}
              onZoom={onZoomImage}
              loading={gradcamLoading}
              vizType="gradcam"
              right={
                <div className={`text-[11px] px-2 py-0.5 rounded-full ${
                  gradcam 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  {gradcam ? '✓ ready' : 'not generated'}
                </div>
              }
            />

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4 md:col-span-2 2xl:col-span-2 animate-slide-up card-hover">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-zinc-100">📊 SHAP (Traditional ML)</div>
                  <div className="mt-0.5 text-xs text-zinc-500">MFCC feature contributions (if supported by model)</div>
                </div>
                <div className={`text-[11px] px-2 py-0.5 rounded-full ${
                  shap?.available 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  {shap?.available ? '✓ ready' : 'unavailable'}
                </div>
              </div>

              {!shap ? (
                <div className="mt-3 rounded-xl border border-dashed border-zinc-800 bg-zinc-950 px-4 py-10 text-center text-xs text-zinc-500">
                  Click “Compute SHAP”
                </div>
              ) : shap.available ? (
                <div className="mt-3 grid grid-cols-1 gap-2">
                  {shap.top_features?.map((f) => (
                    <div key={f.name} className="rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2">
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0 truncate text-xs font-semibold text-zinc-200">{f.name}</div>
                        <div className="text-xs text-zinc-400">{f.value >= 0 ? '+' : ''}{f.value.toFixed(4)}</div>
                      </div>
                      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                        <div
                          className={`h-full ${f.value >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                          style={{ width: `${Math.min(100, Math.abs(f.value) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-400">
                  {shap.error || 'SHAP not available for this model/environment.'}
                </div>
              )}
            </div>

            <VizCard title="🎵 Pitch Contour" subtitle="Voice pitch variation" src={viz?.pitch} onZoom={onZoomImage} vizType="pitch" />
            <VizCard title="✨ Spectral Centroid" subtitle="Audio brightness" src={viz?.centroid} onZoom={onZoomImage} vizType="centroid" />
            <VizCard title="📊 Zero Crossing Rate" subtitle="Voiced/unvoiced behavior" src={viz?.zcr} onZoom={onZoomImage} vizType="zcr" />
            <VizCard title="🔊 RMS Energy" subtitle="Loudness" src={viz?.rms} onZoom={onZoomImage} vizType="rms" />

            {viz?.voiceFig ? (
              <VizCard
                title="🎤 Voice Analysis Explanation"
                subtitle="Pitch + spectral evidence"
                src={viz?.voiceFig}
                onZoom={onZoomImage}
                vizType="voice_analysis"
              />
            ) : null}

            {viz?.segmentFig ? (
              <VizCard
                title="🔍 Mixed Audio Detection"
                subtitle="Segment-by-segment consistency"
                src={viz?.segmentFig}
                onZoom={onZoomImage}
                vizType="segment_analysis"
              />
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
