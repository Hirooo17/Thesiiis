import React from 'react';

export default function Tabs({ tabs, activeKey, onChange }) {
  return (
    <div className="flex items-center gap-1 rounded-xl border border-zinc-800 bg-zinc-900/50 p-1 backdrop-blur-sm">
      {tabs.map((t) => (
        <button
          key={t.key}
          type="button"
          onClick={() => onChange?.(t.key)}
          className={`relative rounded-lg px-4 py-2 text-xs font-semibold transition-all duration-300 ${
            activeKey === t.key
              ? 'bg-gradient-to-r from-blue-500/90 via-purple-500/90 to-pink-500/90 text-white shadow-lg shadow-blue-500/25 scale-[1.02]'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 hover:scale-[1.02]'
          }`}
        >
          {activeKey === t.key && (
            <span className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-pink-400/20 animate-pulse" />
          )}
          <span className="relative">{t.label}</span>
        </button>
      ))}
    </div>
  );
}
