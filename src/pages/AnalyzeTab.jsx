import React from 'react';
import NoisePreviewCard from '../components/NoisePreviewCard.jsx';
import ConfidenceRing from '../components/ConfidenceRing.jsx';
import { clamp, formatPct01 } from '../lib/format.js';

export default function AnalyzeTab({
  references,
  modelType,
  setModelType,
  modelLoaded,
  modelName,
  onLoadDefaultModel,
  onUploadModel,
  recordingDuration,
  setRecordingDuration,
  durationOptions,
  isRecording,
  recordingSeconds,
  onToggleRecording,
  audioLoaded,
  audioUrl,
  onImportAudio,
  snrValue,
  setSnrValue,
  snrLevels,
  customNoiseFile,
  customNoiseUrl,
  onPickCustomNoise,
  onRemoveCustomNoise,
  noiseMixLevel,
  setNoiseMixLevel,
  audioBlobOrFile,
  pushToast,
  canAnalyze,
  onAnalyze,
  canAnalyzeResnetV2,
  onAnalyzeResnetV2,
  onWarmupResnetV2,
  result,
  llmLoading,
  onExplain,
  onWarmupLlm,
  llmExplanation,
  onOpenInfo
}) {
  const verdict = result?.prediction || null;
  const isReal = verdict === 'REAL';

  return (
    <div className="mx-auto grid max-w-[1800px] grid-cols-1 gap-4 px-4 py-5 lg:grid-cols-[360px_1fr] animate-fade-in">
      <aside className="h-fit rounded-2xl border border-zinc-800 bg-zinc-950 p-4 lg:sticky lg:top-24 card-hover">
        <div className="space-y-4 stagger-children">
          <div className="animate-slide-up">
            <div className="text-sm font-semibold text-zinc-100">🤖 Model</div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setModelType('CNN')}
                className={`rounded-xl border px-3 py-2 text-left text-xs font-semibold transition-all duration-300 hover:scale-[1.02] ${
                  modelType === 'CNN'
                    ? 'border-blue-500/40 bg-blue-950/30 text-blue-100 shadow-lg shadow-blue-500/10'
                    : 'border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                <div>CNN (ResNet)</div>
                <div className="mt-0.5 text-[11px] font-normal text-zinc-400">Spectrogram CNN</div>
              </button>
              <button
                type="button"
                onClick={() => setModelType('Traditional')}
                className={`rounded-xl border px-3 py-2 text-left text-xs font-semibold transition-all duration-300 hover:scale-[1.02] ${
                  modelType !== 'CNN'
                    ? 'border-blue-500/40 bg-blue-950/30 text-blue-100 shadow-lg shadow-blue-500/10'
                    : 'border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                <div>Traditional ML</div>
                <div className="mt-0.5 text-[11px] font-normal text-zinc-400">MFCC-based</div>
              </button>
            </div>

            <div className="mt-3 flex flex-col gap-2">
              <button
                type="button"
                onClick={onLoadDefaultModel}
                className="w-full rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 btn-ripple transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/20"
              >
                ⚡ Load Default Model
              </button>

              <label className="w-full cursor-pointer rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-2.5 text-center text-sm font-semibold text-zinc-200 hover:bg-zinc-800 transition-all duration-300 hover:border-zinc-700">
                📂 Upload Model
                <input
                  type="file"
                  accept=".keras,.h5,.joblib,.pkl"
                  className="hidden"
                  onChange={(e) => onUploadModel(e.target.files?.[0] || null)}
                />
              </label>
            </div>
          </div>

          <div className="h-px bg-zinc-800" />

          <div className="animate-slide-up">
            <div className="text-sm font-semibold text-zinc-100">🎤 Audio Input</div>

            <div className="mt-2 space-y-3">
              <div>
                <div className="mb-1 text-xs font-semibold text-zinc-300">Recording Duration</div>
                <div className="flex flex-wrap gap-2">
                  {durationOptions.map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => setRecordingDuration(d)}
                      className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-200 ${
                        recordingDuration === d
                          ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20 scale-105'
                          : 'border border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:scale-105'
                      }`}
                    >
                      {d}s
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button"
                onClick={onToggleRecording}
                className={`w-full rounded-xl px-4 py-3 text-sm font-semibold transition-all duration-300 ${
                  isRecording 
                    ? 'bg-rose-600 text-white hover:bg-rose-500 animate-danger-pulse recording-indicator' 
                    : 'bg-zinc-100 text-zinc-900 hover:bg-white hover:shadow-lg'
                }`}
              >
                {isRecording ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full bg-white animate-pulse" />
                    Stop Recording ({recordingSeconds}s)
                  </span>
                ) : (
                  '🎙️ Record'
                )}
              </button>

              <div className="text-xs text-zinc-400">
                {audioLoaded ? (
                  <span className="text-emerald-300 animate-scale-in inline-flex items-center gap-1">
                    ✅ Audio ready
                  </span>
                ) : (
                  <span>ℹ️ No audio loaded</span>
                )}
              </div>

              {audioUrl ? (
                <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-3 animate-scale-in">
                  <div className="mb-2 text-xs font-semibold text-zinc-300">🔊 Playback</div>
                  <audio src={audioUrl} controls className="w-full" />
                </div>
              ) : null}

              <label className="w-full cursor-pointer rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-2.5 text-center text-sm font-semibold text-zinc-200 hover:bg-zinc-800 transition-all duration-300 hover:border-zinc-700 block">
                📁 Import Audio File
                <input
                  type="file"
                  accept=".wav,.mp3,.flac,.ogg,.m4a,.webm"
                  className="hidden"
                  onChange={(e) => onImportAudio(e.target.files?.[0] || null)}
                />
              </label>
            </div>
          </div>

          <div className="h-px bg-zinc-800" />

          <div className="animate-slide-up">
            <div className="text-sm font-semibold text-zinc-100">🔬 Analyze</div>

            <button
              type="button"
              onClick={onAnalyze}
              disabled={!canAnalyze}
              className={`mt-2 w-full rounded-xl px-4 py-3 text-sm font-semibold transition-all duration-300 ${
                canAnalyze 
                  ? 'bg-emerald-600 text-white hover:bg-emerald-500 hover:shadow-lg hover:shadow-emerald-500/20 btn-ripple' 
                  : 'cursor-not-allowed bg-zinc-800 text-zinc-500'
              }`}
            >
              🚀 Analyze Audio (Local)
            </button>

            <button
              type="button"
              onClick={onAnalyzeResnetV2}
              disabled={!canAnalyzeResnetV2}
              className={`mt-2 w-full rounded-xl px-4 py-3 text-sm font-semibold transition-all duration-300 ${
                canAnalyzeResnetV2
                  ? 'border border-zinc-800 bg-zinc-900 text-zinc-100 hover:bg-zinc-800 hover:border-zinc-700'
                  : 'cursor-not-allowed bg-zinc-800 text-zinc-500'
              }`}
            >
              ⚡ Analyze Audio (ResNet v2)
            </button>

            <button
              type="button"
              onClick={onWarmupResnetV2}
              className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5 text-sm font-semibold text-zinc-200 hover:bg-zinc-900 transition-all duration-300"
            >
              🔥 Warm up ResNet v2
            </button>
          </div>

          <div className="h-px bg-zinc-800" />

          <div className="animate-slide-up">
            <div className="text-sm font-semibold text-zinc-100">🎚️ Add Noise (Optional)</div>

            <div className="mt-2 space-y-3">
              <div>
                <div className="mb-1 text-xs font-semibold text-zinc-300">SNR Noise Simulation</div>
                <select
                  className="w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 transition-all duration-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                  value={snrValue}
                  disabled={Boolean(customNoiseFile)}
                  onChange={(e) => setSnrValue(e.target.value)}
                >
                  <option value="None">None (Original)</option>
                  {snrLevels.map((snr) => (
                    <option key={snr} value={`${snr} dB`}>
                      {snr} dB SNR
                    </option>
                  ))}
                </select>
                <div className="mt-1 text-xs text-zinc-500">Lower dB = more noise</div>
              </div>

              <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-3 transition-all duration-300 hover:border-zinc-700">
                <div className="mb-2 text-xs font-semibold text-zinc-300">🔊 Custom Noise (≤10s)</div>
                <div className="flex flex-col gap-2">
                  <label className="cursor-pointer rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-center text-sm font-semibold text-zinc-200 hover:bg-zinc-900 transition-all duration-200">
                    Upload Noise File
                    <input
                      type="file"
                      accept=".wav,.mp3,.flac,.ogg,.m4a,.webm"
                      className="hidden"
                      onChange={(e) => onPickCustomNoise(e.target.files?.[0] || null)}
                    />
                  </label>

                  {customNoiseFile ? (
                    <div className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 animate-scale-in">
                      <div className="min-w-0 truncate text-sm text-zinc-200">{customNoiseFile.name}</div>
                      <button
                        type="button"
                        onClick={onRemoveCustomNoise}
                        className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 hover:bg-zinc-800 transition-colors"
                      >
                        Remove
                      </button>
                    </div>
                  ) : null}

                  {customNoiseUrl ? <audio src={customNoiseUrl} controls className="w-full" /> : null}

                  <div className={`space-y-1 transition-opacity duration-300 ${customNoiseFile ? '' : 'opacity-50'}`}>
                    <div className="flex items-center justify-between text-xs text-zinc-400">
                      <span>Noise Mix Level</span>
                      <span>{noiseMixLevel}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={noiseMixLevel}
                      disabled={!customNoiseFile}
                      onChange={(e) => setNoiseMixLevel(clamp(Number(e.target.value), 0, 100))}
                      className="w-full accent-blue-500"
                    />
                  </div>
                </div>
              </div>

              <NoisePreviewCard
                audioBlobOrFile={audioBlobOrFile}
                snrValue={snrValue}
                customNoiseFile={customNoiseFile}
                noiseMixLevel={noiseMixLevel}
                onToast={pushToast}
              />
            </div>
          </div>

          <div className="h-px bg-zinc-800" />

          <div className="animate-slide-up">
            <div className="text-sm font-semibold text-zinc-100">📊 Verdict</div>
            <div className={`mt-2 rounded-2xl border bg-gradient-to-br from-zinc-900 to-zinc-950 p-4 transition-all duration-500 ${
              verdict 
                ? isReal 
                  ? 'border-emerald-500/30 shadow-lg shadow-emerald-500/10' 
                  : 'border-rose-500/30 shadow-lg shadow-rose-500/10'
                : 'border-zinc-800'
            }`}>
              {verdict ? (
                <div className="flex items-center gap-4 animate-scale-in">
                  <ConfidenceRing value={result?.confidence || 0} size={80} isReal={isReal} />
                  <div className="flex-1">
                    <div className={`text-xl font-bold ${isReal ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {isReal ? '✅ REAL' : '❌ FAKE'}
                    </div>
                    <div className="mt-1 text-xs text-zinc-400">
                      {result?.audio_duration ? `${result.audio_duration.toFixed(2)}s audio` : ''}
                      {typeof result?.adjusted_confidence === 'number' ? ` • Adj: ${formatPct01(result.adjusted_confidence)}` : ''}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-sm text-zinc-400">
                  <div className="text-2xl mb-2 animate-float">🎧</div>
                  Awaiting analysis…
                </div>
              )}

              {result ? (
                <button
                  type="button"
                  onClick={onExplain}
                  disabled={llmLoading}
                  className={`mt-4 w-full rounded-xl px-4 py-2.5 text-sm font-semibold transition-all duration-300 ${
                    llmLoading 
                      ? 'cursor-wait bg-purple-900 text-purple-300' 
                      : 'bg-purple-600 text-white hover:bg-purple-500 hover:shadow-lg hover:shadow-purple-500/20 btn-ripple'
                  }`}
                >
                  {llmLoading ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="inline-block h-4 w-4 border-2 border-purple-300 border-t-transparent rounded-full animate-spin" />
                      Generating… (30–60s)
                    </span>
                  ) : (
                    '🧠 Explain Results (AI)'
                  )}
                </button>
              ) : null}

              <button
                type="button"
                onClick={onWarmupLlm}
                disabled={llmLoading}
                className={`mt-2 w-full rounded-xl border border-purple-500/30 bg-purple-950/30 px-4 py-2 text-xs font-semibold transition-all duration-300 ${
                  llmLoading ? 'cursor-wait text-purple-400' : 'text-purple-300 hover:bg-purple-900/30'
                }`}
              >
                {llmLoading ? '⏳ Loading AI…' : '🔥 Warm up AI (before demo)'}
              </button>

              {llmExplanation ? (
                <div className="mt-4 rounded-xl border border-purple-500/30 bg-purple-950/30 p-3 animate-scale-in">
                  <div className="mb-2 text-xs font-semibold text-purple-300">🤖 AI Explanation</div>
                  <div className="text-sm leading-relaxed text-zinc-200">{llmExplanation}</div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </aside>

      <main className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4 card-hover animate-slide-up">
        <div className="mb-4 flex items-end justify-between gap-3">
          <div>
            <div className="text-base font-semibold text-zinc-100">👁️ Quick View</div>
            <div className="text-xs text-zinc-500">Run analysis, then open Explainability tab.</div>
          </div>

          <button
            type="button"
            onClick={() => onOpenInfo?.('spectrogram')}
            className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-zinc-800 transition-all duration-200 hover:scale-105"
          >
            ❓ What do these mean?
          </button>
        </div>

        {!result ? (
          <div className="grid place-items-center rounded-2xl border border-dashed border-zinc-800 bg-zinc-950 py-20 text-center">
            <div className="max-w-md space-y-2">
              <div className="text-4xl animate-float">🎵</div>
              <div className="text-lg font-semibold text-zinc-100">Ready when you are</div>
              <div className="text-sm text-zinc-400">Load a model + audio, then click Analyze.</div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 stagger-children">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4 animate-slide-up card-hover">
              <div className="text-sm font-semibold text-zinc-100">🤖 Model</div>
              <div className="mt-1 text-xs text-zinc-400">{modelName}</div>
              <div className="mt-3 text-xs text-zinc-300">
                Verdict: <span className={`font-semibold ${isReal ? 'text-emerald-300' : 'text-rose-300'}`}>{verdict}</span>
              </div>
            </div>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4 animate-slide-up card-hover">
              <div className="text-sm font-semibold text-zinc-100">📝 Notes</div>
              <div className="mt-2 text-xs text-zinc-300">
                {result?.mixed_audio_warning ? result.mixed_audio_warning : 'No mixed-audio warning.'}
              </div>
              <div className="mt-2 text-xs text-zinc-400">
                Noise impact: <span className="font-semibold">{String(result?.noise_analysis?.impact_level || '--').toUpperCase()}</span>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
