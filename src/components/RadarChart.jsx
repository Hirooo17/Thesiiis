import React, { useMemo, useState, useEffect } from 'react';

/**
 * A simple radar/spider chart for visualizing technical metrics.
 * @param {{ metrics: { label: string, value: number }[], size?: number }} props
 * value should be normalized 0-1
 */
export default function RadarChart({ metrics = [], size = 200 }) {
  const [animated, setAnimated] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(timer);
  }, []);
  
  const center = size / 2;
  const maxRadius = size / 2 - 25;
  const angleStep = (2 * Math.PI) / metrics.length;

  const gradientId = `radar-gradient-${Math.random().toString(36).slice(2)}`;
  const glowId = `radar-glow-${Math.random().toString(36).slice(2)}`;

  const points = useMemo(() => {
    return metrics.map((m, i) => {
      const angle = i * angleStep - Math.PI / 2; // Start from top
      const r = (animated ? m.value : 0) * maxRadius;
      return {
        x: center + r * Math.cos(angle),
        y: center + r * Math.sin(angle),
        labelX: center + (maxRadius + 18) * Math.cos(angle),
        labelY: center + (maxRadius + 18) * Math.sin(angle),
        label: m.label,
        value: m.value
      };
    });
  }, [metrics, angleStep, center, maxRadius, animated]);

  // Grid lines (concentric polygons at 25%, 50%, 75%, 100%)
  const gridLevels = [0.25, 0.5, 0.75, 1];
  const gridPolygons = gridLevels.map((level) => {
    const pts = metrics.map((_, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const r = level * maxRadius;
      return `${center + r * Math.cos(angle)},${center + r * Math.sin(angle)}`;
    });
    return pts.join(' ');
  });

  // Axis lines
  const axisLines = metrics.map((_, i) => {
    const angle = i * angleStep - Math.PI / 2;
    return {
      x2: center + maxRadius * Math.cos(angle),
      y2: center + maxRadius * Math.sin(angle)
    };
  });

  // Data polygon
  const dataPolygon = points.map((p) => `${p.x},${p.y}`).join(' ');

  if (metrics.length < 3) {
    return (
      <div className="flex items-center justify-center text-xs text-zinc-500" style={{ width: size, height: size }}>
        Need at least 3 metrics
      </div>
    );
  }

  return (
    <svg width={size} height={size} className="overflow-visible">
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.5" />
          <stop offset="50%" stopColor="#8B5CF6" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#EC4899" stopOpacity="0.2" />
        </linearGradient>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>

      {/* Grid polygons */}
      {gridPolygons.map((pts, i) => (
        <polygon
          key={i}
          points={pts}
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          className="text-zinc-800"
          strokeDasharray={i < gridLevels.length - 1 ? '2,4' : 'none'}
        />
      ))}

      {/* Axis lines */}
      {axisLines.map((line, i) => (
        <line
          key={i}
          x1={center}
          y1={center}
          x2={line.x2}
          y2={line.y2}
          stroke="currentColor"
          strokeWidth="1"
          className="text-zinc-700"
        />
      ))}

      {/* Data polygon with gradient fill */}
      <polygon
        points={dataPolygon}
        fill={`url(#${gradientId})`}
        stroke="url(#radar-stroke)"
        strokeWidth="2.5"
        filter={`url(#${glowId})`}
        className="transition-all duration-1000 ease-out"
        style={{ stroke: '#3B82F6' }}
      />

      {/* Data points */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r="5"
          fill="#3B82F6"
          stroke="#1E3A5F"
          strokeWidth="2"
          className="transition-all duration-1000 ease-out"
          style={{
            filter: 'drop-shadow(0 0 4px rgba(59, 130, 246, 0.5))'
          }}
        />
      ))}

      {/* Labels */}
      {points.map((p, i) => (
        <text
          key={i}
          x={p.labelX}
          y={p.labelY}
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-zinc-400 text-[9px] font-medium"
        >
          {p.label}
        </text>
      ))}
    </svg>
  );
}
