export function pickRecorderMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported?.(c)) return c;
  }
  return '';
}

function clamp01(x) {
  return Math.max(-1, Math.min(1, x));
}

export async function decodeAudioToMonoFloat32(blobOrFile) {
  const arrayBuffer = await blobOrFile.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const decoded = await audioContext.decodeAudioData(arrayBuffer);
    const { numberOfChannels, length, sampleRate } = decoded;

    // Downmix to mono
    const mono = new Float32Array(length);
    for (let ch = 0; ch < numberOfChannels; ch++) {
      const data = decoded.getChannelData(ch);
      for (let i = 0; i < length; i++) mono[i] += data[i] / numberOfChannels;
    }

    return { samples: mono, sampleRate };
  } finally {
    audioContext.close?.();
  }
}

function meanPower(x) {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  return x.length ? sum / x.length : 0;
}

function randn() {
  // Box–Muller transform
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function applySnrWhiteNoise(samples, snrDb) {
  if (!Number.isFinite(snrDb)) return samples;
  const signalPower = meanPower(samples);
  if (signalPower <= 0) return samples;

  const snrLinear = Math.pow(10, snrDb / 10);
  const noisePowerTarget = signalPower / snrLinear;
  const noiseStd = Math.sqrt(noisePowerTarget);

  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const n = randn() * noiseStd;
    out[i] = clamp01(samples[i] + n);
  }
  return out;
}

export function mixWithNoiseTrack(samples, noiseSamples, mixPercent) {
  if (!noiseSamples?.length) return samples;
  const mix = Math.max(0, Math.min(100, mixPercent)) / 100;
  if (mix === 0) return samples;

  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const n = noiseSamples[i % noiseSamples.length];
    out[i] = clamp01(samples[i] * (1 - mix) + n * mix);
  }
  return out;
}

export function encodeWav16(samples, sampleRate) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeAscii = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM
  view.setUint16(20, 1, true); // format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits
  writeAscii(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
