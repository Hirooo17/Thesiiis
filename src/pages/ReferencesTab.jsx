import React, { useMemo, useState } from 'react';

function LegendCard({ title, bullets, refs, open, onToggle }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-zinc-100">{title}</div>
          <div className="mt-0.5 text-xs text-zinc-500">Click to expand</div>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-zinc-800"
        >
          {open ? 'Hide' : 'Show'}
        </button>
      </div>

      {open ? (
        <div className="mt-3 space-y-3">
          {bullets?.length ? (
            <ul className="list-disc space-y-1 pl-5 text-sm text-zinc-200">
              {bullets.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          ) : null}

          {refs?.length ? (
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Research References</div>
              <ul className="space-y-1">
                {refs.map((r) => (
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
      ) : null}
    </div>
  );
}

export default function ReferencesTab({ references }) {
  const [openLegend, setOpenLegend] = useState('waveform');

  const info = useMemo(() => {
    const listRefs = (key) => (references?.[key] || []).map((r) => ({ name: r.name, url: r.url }));

    return [
      {
        key: 'waveform',
        title: 'Waveform',
        bullets: ['Tall waves = loud', 'Flat = silence', 'Regular patterns = consistent voice'],
        refs: listRefs('waveform')
      },
      {
        key: 'spectrogram',
        title: 'Mel Spectrogram',
        bullets: ['Audio “fingerprint” over time', 'Bright = strong frequency content', 'AI may show artifacts'],
        refs: listRefs('spectrogram')
      },
      {
        key: 'saliency',
        title: 'Saliency Heatmap',
        bullets: ['Shows where model focused', 'Red = important', 'Used for explainability'],
        refs: listRefs('saliency')
      },
      {
        key: 'confidence',
        title: 'Confidence Score',
        bullets: ['80%+ very confident', '60–80% moderate', '<60% uncertain'],
        refs: listRefs('confidence')
      },
      {
        key: 'voice_analysis',
        title: 'Voice Analysis',
        bullets: ['Single voice = normal', 'Multiple voices = possible mix', 'Uses pitch & spectral analysis'],
        refs: listRefs('voice_analysis')
      },
      {
        key: 'noise',
        title: 'Noise & SNR',
        bullets: ['Higher SNR = cleaner', 'Noise can mask artifacts', 'Model robustness depends on noise'],
        refs: listRefs('noise')
      },
      {
        key: 'resnet',
        title: 'ResNet Architecture',
        bullets: ['Deep CNN with residual connections', 'Transfer learning friendly', 'Works well on spectrogram images'],
        refs: listRefs('resnet')
      },
      {
        key: 'real_verdict',
        title: 'REAL Verdict',
        bullets: ['Natural pitch variations', 'Realistic micro-variations', 'Less synthetic artifacts'],
        refs: listRefs('real_verdict')
      },
      {
        key: 'fake_verdict',
        title: 'FAKE Verdict',
        bullets: ['Unnatural smoothness', 'Spectral artifacts', 'Missing human micro-expressions'],
        refs: listRefs('fake_verdict')
      }
    ];
  }, [references]);

  return (
    <div className="mx-auto max-w-[1800px] px-4 py-5">
      <div className="mb-4">
        <div className="text-base font-semibold text-zinc-100">Understanding Results</div>
        <div className="text-xs text-zinc-500">Explanations + research references</div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3">
        {info.map((c) => (
          <LegendCard
            key={c.key}
            title={c.title}
            bullets={c.bullets}
            refs={c.refs}
            open={openLegend === c.key}
            onToggle={() => setOpenLegend(openLegend === c.key ? '' : c.key)}
          />
        ))}
      </div>
    </div>
  );
}
