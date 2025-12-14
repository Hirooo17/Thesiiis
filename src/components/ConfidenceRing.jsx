import React, { useState, useEffect } from 'react';

/**
 * A circular confidence gauge with animated ring.
 * @param {{ value: number, size?: number, strokeWidth?: number, isReal?: boolean }} props
 */
export default function ConfidenceRing({ value = 0, size = 100, strokeWidth = 8, isReal = true }) {
  const [animatedValue, setAnimatedValue] = useState(0);
  
  // Animate the value on mount and when it changes
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValue(value);
    }, 100);
    return () => clearTimeout(timer);
  }, [value]);
  
  const normalizedValue = Math.max(0, Math.min(1, animatedValue));
  const percentage = Math.round(normalizedValue * 100);
  
  const center = size / 2;
  const radius = center - strokeWidth / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - normalizedValue);
  
  const gradientId = `confidence-gradient-${Math.random().toString(36).slice(2)}`;
  const glowId = `confidence-glow-${Math.random().toString(36).slice(2)}`;

  return (
    <div className="relative inline-flex items-center justify-center animate-scale-in">
      <svg width={size} height={size} className="transform -rotate-90">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            {isReal ? (
              <>
                <stop offset="0%" stopColor="#10B981" />
                <stop offset="50%" stopColor="#34D399" />
                <stop offset="100%" stopColor="#6EE7B7" />
              </>
            ) : (
              <>
                <stop offset="0%" stopColor="#F43F5E" />
                <stop offset="50%" stopColor="#FB7185" />
                <stop offset="100%" stopColor="#FDA4AF" />
              </>
            )}
          </linearGradient>
          <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        {/* Background ring */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          className="text-zinc-800"
          stroke="currentColor"
        />
        {/* Progress ring with gradient */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          stroke={`url(#${gradientId})`}
          filter={`url(#${glowId})`}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-lg font-bold transition-all duration-500 ${isReal ? 'text-emerald-400' : 'text-rose-400'}`}>
          {percentage}%
        </span>
        <span className="text-[10px] text-zinc-500">confidence</span>
      </div>
    </div>
  );
}
