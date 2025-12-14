import React, { useEffect, useState } from 'react';

export default function Lightbox({ open, src, alt, onClose, vizType, result }) {
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onClose?.();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  // Fetch explanation when lightbox opens
  useEffect(() => {
    if (!open || !vizType) {
      setExplanation(null);
      setExpanded(false);
      return;
    }

    const fetchExplanation = async () => {
      setLoading(true);
      try {
        const res = await fetch('/api/explain_visualization', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            viz_type: vizType,
            prediction: result?.prediction || 'unknown',
            confidence: result?.confidence || 0,
            audio_duration: result?.audio_duration || 0,
            pitch_std: result?.voice_analysis?.pitch_std || 0,
            spectral_variation: result?.voice_analysis?.spectral_variation || 0,
            snr: result?.noise_analysis?.snr_estimated || 0
          })
        });
        const data = await res.json();
        if (data.success) {
          setExplanation(data.explanation);
        }
      } catch (err) {
        console.error('Failed to fetch explanation:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchExplanation();
  }, [open, vizType, result]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 p-4 animate-fade-in"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-6xl overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 animate-scale-in">
        <div className="flex items-center justify-between gap-4 border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-3">
            <span className="text-xl">📊</span>
            <div className="truncate text-sm font-semibold text-zinc-200">{alt || 'Visualization'}</div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className={`rounded-lg border px-3 py-1.5 text-sm transition-all duration-200 ${
                expanded 
                  ? 'border-purple-500/40 bg-purple-950/30 text-purple-300' 
                  : 'border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800'
              }`}
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? '📖 Hide Explanation' : '🧠 Explain This'}
            </button>
            <button
              type="button"
              className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800 transition-colors"
              onClick={onClose}
            >
              ✕ Close
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_350px]">
          {/* Image */}
          <div className="max-h-[70vh] overflow-auto p-4 border-r border-zinc-800">
            <img 
              src={src} 
              alt={alt || 'Visualization'} 
              className="h-auto w-full rounded-xl transition-transform duration-300 hover:scale-[1.02]" 
            />
          </div>
          
          {/* Explanation Panel */}
          <div className={`transition-all duration-300 overflow-hidden ${expanded ? 'block' : 'hidden lg:block'}`}>
            <div className="p-4 h-full">
              <div className="rounded-xl border border-purple-500/30 bg-purple-950/20 p-4 h-full">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-lg">🤖</span>
                  <div className="text-sm font-semibold text-purple-300">AI Explanation</div>
                </div>
                
                {loading ? (
                  <div className="flex items-center gap-3 text-zinc-400">
                    <div className="h-5 w-5 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm">Generating explanation...</span>
                  </div>
                ) : explanation ? (
                  <div className="space-y-3">
                    <div className="text-sm leading-relaxed text-zinc-300">
                      {explanation}
                    </div>
                    
                    {result && (
                      <div className="mt-4 pt-4 border-t border-purple-500/20">
                        <div className="text-xs font-semibold text-purple-400 mb-2">Detection Summary</div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="rounded-lg bg-zinc-900 px-3 py-2">
                            <div className="text-zinc-500">Verdict</div>
                            <div className={`font-semibold ${result.prediction === 'REAL' ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {result.prediction === 'REAL' ? '✅ REAL' : '❌ FAKE'}
                            </div>
                          </div>
                          <div className="rounded-lg bg-zinc-900 px-3 py-2">
                            <div className="text-zinc-500">Confidence</div>
                            <div className="font-semibold text-zinc-200">{(result.confidence * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-sm text-zinc-500">
                    Click "Explain This" to get an AI-powered explanation of this visualization.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
