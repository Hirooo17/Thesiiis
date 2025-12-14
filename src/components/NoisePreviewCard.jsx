import React, { useEffect, useMemo, useState } from 'react';
import {
  applySnrWhiteNoise,
  decodeAudioToMonoFloat32,
  encodeWav16,
  mixWithNoiseTrack
} from '../lib/audio.js';

function parseSnrDb(snrValue) {
  if (!snrValue || snrValue === 'None') return null;
  const m = String(snrValue).match(/(\d+(?:\.\d+)?)/);
  if (!m) return null;
  return Number(m[1]);
}

export default function NoisePreviewCard({
  audioBlobOrFile,
  snrValue,
  customNoiseFile,
  noiseMixLevel,
  onToast
}) {
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  const [originalUrl, setOriginalUrl] = useState('');

  const snrDb = useMemo(() => parseSnrDb(snrValue), [snrValue]);

  useEffect(() => {
    if (!audioBlobOrFile) return;
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    const url = URL.createObjectURL(audioBlobOrFile);
    setOriginalUrl(url);
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioBlobOrFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const canPreview = Boolean(audioBlobOrFile) && (snrDb !== null || Boolean(customNoiseFile));

  async function generatePreview() {
    if (!canPreview) return;

    setBusy(true);
    try {
      const { samples, sampleRate } = await decodeAudioToMonoFloat32(audioBlobOrFile);

      let out = samples;

      if (snrDb !== null) {
        out = applySnrWhiteNoise(out, snrDb);
      }

      if (customNoiseFile) {
        const decodedNoise = await decodeAudioToMonoFloat32(customNoiseFile);
        out = mixWithNoiseTrack(out, decodedNoise.samples, noiseMixLevel);
      }

      const wavBlob = encodeWav16(out, sampleRate);
      const url = URL.createObjectURL(wavBlob);

      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(url);

      onToast?.('Noise preview generated', 'success');
    } catch (e) {
      onToast?.(e?.message || 'Failed to generate noise preview', 'error');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold text-zinc-300">Noise Preview</div>
          <div className="mt-0.5 text-[11px] text-zinc-500">
            Generate a playable preview of the injected noise (SNR and/or custom noise mix).
          </div>
        </div>
        <button
          type="button"
          onClick={generatePreview}
          disabled={!canPreview || busy}
          className={`shrink-0 rounded-lg px-3 py-1.5 text-xs font-semibold ${
            !canPreview || busy
              ? 'cursor-not-allowed bg-zinc-800 text-zinc-500'
              : 'border border-zinc-800 bg-zinc-950 text-zinc-200 hover:bg-zinc-800'
          }`}
        >
          {busy ? 'Generating…' : 'Generate Preview'}
        </button>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Original</div>
          {originalUrl ? <audio src={originalUrl} controls className="w-full" /> : <div className="text-xs text-zinc-500">No audio</div>}
        </div>

        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">With Noise</div>
          {previewUrl ? (
            <audio src={previewUrl} controls className="w-full" />
          ) : (
            <div className="text-xs text-zinc-500">Click “Generate Preview”</div>
          )}
        </div>
      </div>
    </div>
  );
}
