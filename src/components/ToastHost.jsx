import React, { useEffect } from 'react';

export function ToastHost({ toasts, onDismiss }) {
  useEffect(() => {
    const timers = toasts.map((t) =>
      setTimeout(() => {
        onDismiss?.(t.id);
      }, t.durationMs ?? 3200)
    );
    return () => timers.forEach(clearTimeout);
  }, [toasts, onDismiss]);

  return (
    <div className="fixed right-4 top-20 z-[60] flex w-[min(92vw,420px)] flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-start gap-3 rounded-xl border px-4 py-3 shadow-lg backdrop-blur ${
            t.type === 'success'
              ? 'border-emerald-900/60 bg-emerald-950/60 text-emerald-50'
              : t.type === 'error'
              ? 'border-rose-900/60 bg-rose-950/60 text-rose-50'
              : 'border-blue-900/60 bg-blue-950/60 text-blue-50'
          }`}
        >
          <div className="min-w-0 flex-1 text-sm">{t.message}</div>
          <button
            type="button"
            className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs hover:bg-white/10"
            onClick={() => onDismiss?.(t.id)}
          >
            Dismiss
          </button>
        </div>
      ))}
    </div>
  );
}
