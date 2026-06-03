import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  data: any;
  currentFrame: number;
}

type ViewMode = 'live' | 'heatmap';

const COURT = { width: 10.97, length: 23.77, singles_width: 8.23, service_line: 6.4 };
const SCALE = 16;
const PAD_X = 28;
const PAD_Y = 36;
const CW = COURT.width * SCALE + PAD_X * 2;
const CH = COURT.length * SCALE + PAD_Y * 2;
const MARGIN = (COURT.width - COURT.singles_width) / 2;

const px = (x: number) => x * SCALE + PAD_X;
const py = (y: number) => y * SCALE + PAD_Y;

const P1 = '#ff6060';
const P2 = '#4d9eff';

// Grid for heatmap: 8 cols × 14 rows
const GRID_COLS = 8;
const GRID_ROWS = 14;

const Court2D: React.FC<Props> = ({ data, currentFrame }) => {
  const [mode, setMode] = useState<ViewMode>('live');
  const playerTracks = data.results.player_metric_tracks || { top: [], bottom: [] };
  const bounces: any[] = data.results.bounces || [];

  // Sorted once for O(n) lookup. Bounce dots use next-bounce coloring:
  // the dot at frame F gets the color of whoever caused the bounce at F.
  const ballSpeeds: any[] = useMemo(
    () => [...(data.results.analytics?.ball_speeds ?? [])].sort((a, b) => a.end - b.end),
    [data]
  );

  const getHitterColor = (bounceFrame: number): string => {
    // The bounce's own event has end === bounceFrame (or very close).
    // Find it by matching end frame directly, then fall back to next-event.
    const exact = ballSpeeds.find((s) => s.end === bounceFrame);
    if (exact) return exact.side === 'top' ? P1 : P2;

    // Fallback: first event whose end >= bounceFrame
    const next = ballSpeeds.find((s) => s.end >= bounceFrame);
    if (next) return next.side === 'top' ? P1 : P2;

    if (ballSpeeds.length > 0) {
      return ballSpeeds[ballSpeeds.length - 1].side === 'top' ? P1 : P2;
    }
    return P1;
  };

  // Build heatmap grid
  const heatmapGrid = useMemo(() => {
    const cells: number[][] = Array.from({ length: GRID_ROWS }, () => new Array(GRID_COLS).fill(0));
    let maxVal = 0;
    bounces.forEach((b) => {
      if (!b.pos_2d || b.status.includes('Out')) return;
      const [x, y] = b.pos_2d;
      const col = Math.min(GRID_COLS - 1, Math.max(0, Math.floor((x / COURT.width) * GRID_COLS)));
      const row = Math.min(GRID_ROWS - 1, Math.max(0, Math.floor((y / COURT.length) * GRID_ROWS)));
      cells[row][col]++;
      if (cells[row][col] > maxVal) maxVal = cells[row][col];
    });
    return { cells, maxVal };
  }, [bounces]);

  const inCount = bounces.filter((b) => b.status === 'In').length;
  const outCount = bounces.filter((b) => b.status.includes('Out')).length;

  return (
    <div
      className="rounded-2xl overflow-hidden"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-4 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-white text-sm">Court Map</h3>
          <p className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
            {inCount} In · {outCount} Out
          </p>
        </div>
        {/* Mode toggle */}
        <div
          className="flex rounded-xl p-0.5 gap-0.5"
          style={{ background: 'rgba(255,255,255,0.05)' }}
        >
          {(['live', 'heatmap'] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className="relative px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors"
              style={{ color: mode === m ? 'black' : 'rgba(255,255,255,0.4)' }}
            >
              {mode === m && (
                <motion.div
                  layoutId="modePill"
                  className="absolute inset-0 rounded-lg"
                  style={{ background: '#f5c518' }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <span className="relative capitalize">{m}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Court SVG */}
      <div className="flex justify-center pb-5 px-4">
        <svg
          width={CW}
          height={CH}
          viewBox={`0 0 ${CW} ${CH}`}
          className="rounded-xl"
          style={{ background: '#1e4d1a', maxWidth: '100%' }}
        >
          <defs>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="1.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>
          {/* Court surface */}
          <rect x={PAD_X} y={PAD_Y} width={COURT.width * SCALE} height={COURT.length * SCALE} fill="#2d6e28" />

          {/* Heatmap overlay */}
          <AnimatePresence>
            {mode === 'heatmap' && heatmapGrid.maxVal > 0 && (
              <g key="heatmap">
                {heatmapGrid.cells.map((row, ri) =>
                  row.map((val, ci) => {
                    if (val === 0) return null;
                    const intensity = val / heatmapGrid.maxVal;
                    const cellW = (COURT.width * SCALE) / GRID_COLS;
                    const cellH = (COURT.length * SCALE) / GRID_ROWS;
                    const cx = PAD_X + ci * cellW;
                    const cy = PAD_Y + ri * cellH;
                    return (
                      <motion.rect
                        key={`h-${ri}-${ci}`}
                        x={cx} y={cy}
                        width={cellW} height={cellH}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: intensity * 0.85 }}
                        exit={{ opacity: 0 }}
                        style={{
                          fill: `rgba(245, 80, 50, 1)`,
                        }}
                      />
                    );
                  })
                )}
              </g>
            )}
          </AnimatePresence>

          {/* Court lines */}
          {/* Outer boundary */}
          <rect x={px(0)} y={py(0)} width={COURT.width * SCALE} height={COURT.length * SCALE}
            fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="2" />
          {/* Singles sidelines */}
          <line x1={px(MARGIN)} y1={py(0)} x2={px(MARGIN)} y2={py(COURT.length)} stroke="rgba(255,255,255,0.7)" strokeWidth="1" />
          <line x1={px(COURT.width - MARGIN)} y1={py(0)} x2={px(COURT.width - MARGIN)} y2={py(COURT.length)} stroke="rgba(255,255,255,0.7)" strokeWidth="1" />
          {/* Service lines */}
          <line x1={px(MARGIN)} y1={py(COURT.length / 2 - COURT.service_line)} x2={px(COURT.width - MARGIN)} y2={py(COURT.length / 2 - COURT.service_line)} stroke="rgba(255,255,255,0.7)" strokeWidth="1" />
          <line x1={px(MARGIN)} y1={py(COURT.length / 2 + COURT.service_line)} x2={px(COURT.width - MARGIN)} y2={py(COURT.length / 2 + COURT.service_line)} stroke="rgba(255,255,255,0.7)" strokeWidth="1" />
          {/* Center service line */}
          <line x1={px(COURT.width / 2)} y1={py(COURT.length / 2 - COURT.service_line)} x2={px(COURT.width / 2)} y2={py(COURT.length / 2 + COURT.service_line)} stroke="rgba(255,255,255,0.7)" strokeWidth="1" />
          {/* Net */}
          <line x1={px(-0.3)} y1={py(COURT.length / 2)} x2={px(COURT.width + 0.3)} y2={py(COURT.length / 2)} stroke="rgba(0,0,0,0.9)" strokeWidth="4" />
          <line x1={px(-0.3)} y1={py(COURT.length / 2)} x2={px(COURT.width + 0.3)} y2={py(COURT.length / 2)} stroke="rgba(255,255,255,0.3)" strokeWidth="1" strokeDasharray="4,3" />

          {/* Player zone labels */}
          <text x={px(COURT.width / 2)} y={py(COURT.length * 0.12)} textAnchor="middle" fill={P1} fontSize="9" fontWeight="700" opacity="0.6">P1</text>
          <text x={px(COURT.width / 2)} y={py(COURT.length * 0.91)} textAnchor="middle" fill={P2} fontSize="9" fontWeight="700" opacity="0.6">P2</text>

          {/* Live mode content */}
          {mode === 'live' && (
            <>
              {/* Player paths (Segmented for fade effect) */}
              {(['top', 'bottom'] as const).map((side) => {
                const track = playerTracks[side];
                const window = 3;
                const trailLength = 40;
                const segments = [];
                
                for (let i = Math.max(1, currentFrame - trailLength); i <= currentFrame; i++) {
                  const getPos = (idx: number) => {
                    let sx = 0, sy = 0, cnt = 0;
                    for (let j = Math.max(0, idx - window); j <= Math.min(track.length - 1, idx + window); j++) {
                      const p = track[j];
                      if (p?.x !== null && p?.x !== undefined) { sx += p.x; sy += p.y; cnt++; }
                    }
                    return cnt > 0 ? { x: px(sx / cnt), y: py(sy / cnt) } : null;
                  };

                  const p1 = getPos(i - 1);
                  const p2 = getPos(i);

                  if (p1 && p2) {
                    const age = currentFrame - i;
                    const opacity = Math.max(0, 0.4 * (1 - age / trailLength));
                    segments.push(
                      <line
                        key={`${side}-${i}`}
                        x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                        stroke={side === 'top' ? P1 : P2}
                        strokeWidth={2.5 * (1 - age / trailLength)}
                        strokeOpacity={opacity}
                        strokeLinecap="round"
                        filter="url(#glow)"
                      />
                    );
                  }
                }
                return <g key={`path-${side}`}>{segments}</g>;
              })}

              {/* Player dots */}
              {(['top', 'bottom'] as const).map((side) => {
                const pt = playerTracks[side][currentFrame];
                if (!pt || pt.x == null) return null;
                const color = side === 'top' ? P1 : P2;
                return (
                  <g key={`dot-${side}`}>
                    {/* Pulsing aura */}
                    <motion.circle
                      cx={px(pt.x)}
                      cy={py(pt.y)}
                      r="12"
                      fill={color}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: [0.05, 0.15, 0.05], scale: [1, 1.5, 1] }}
                      transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                    />
                    <motion.circle
                      r="6"
                      fill={color}
                      stroke="white"
                      strokeWidth="2"
                      filter="url(#glow)"
                      animate={{ cx: px(pt.x), cy: py(pt.y) }}
                      transition={{ type: 'spring', damping: 18, stiffness: 130 }}
                    />
                  </g>
                );
              })}

              {/* Bounce dots */}
              {bounces.map((bounce, idx) => {
                if (!bounce.pos_2d) return null;
                const isPast = bounce.frame <= currentFrame;
                const isFresh = Math.abs(bounce.frame - currentFrame) < 5;
                const isNear = Math.abs(bounce.frame - currentFrame) < 30;
                const isOut = bounce.status.includes('Out');
                const col = getHitterColor(bounce.frame);
                
                if (!isPast) return null;

                return (
                  <g key={`bounce-${idx}`}>
                    {/* Multi-layered dynamic shockwave when ball drops */}
                    {isFresh && (
                      <>
                        <motion.circle
                          cx={px(bounce.pos_2d[0])}
                          cy={py(bounce.pos_2d[1])}
                          initial={{ r: 0, opacity: 0.8, strokeWidth: 2 }}
                          animate={{ r: 35, opacity: 0, strokeWidth: 0 }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          fill="none"
                          stroke={col}
                        />
                        <motion.circle
                          cx={px(bounce.pos_2d[0])}
                          cy={py(bounce.pos_2d[1])}
                          initial={{ r: 0, opacity: 1, strokeWidth: 1 }}
                          animate={{ r: 20, opacity: 0, strokeWidth: 0 }}
                          transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
                          fill="none"
                          stroke="white"
                        />
                      </>
                    )}
                    
                    <motion.g
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ 
                        scale: isFresh ? [0, 1.5, 1] : 1,
                        opacity: isNear ? 1 : 0.35 
                      }}
                      transition={{ duration: 0.5, type: "spring", damping: 15 }}
                    >
                      {isOut ? (
                        <g filter="url(#glow)">
                          <circle cx={px(bounce.pos_2d[0])} cy={py(bounce.pos_2d[1])} r={isNear ? 6 : 4} fill="none" stroke={col} strokeWidth="2" />
                          <line x1={px(bounce.pos_2d[0]) - 3} y1={py(bounce.pos_2d[1]) - 3} x2={px(bounce.pos_2d[0]) + 3} y2={py(bounce.pos_2d[1]) + 3} stroke={col} strokeWidth="2" />
                          <line x1={px(bounce.pos_2d[0]) + 3} y1={py(bounce.pos_2d[1]) - 3} x2={px(bounce.pos_2d[0]) - 3} y2={py(bounce.pos_2d[1]) + 3} stroke={col} strokeWidth="2" />
                        </g>
                      ) : (
                        <circle cx={px(bounce.pos_2d[0])} cy={py(bounce.pos_2d[1])} r={isNear ? 5.5 : 3.5} fill={col} stroke="white" strokeWidth="1.5" filter="url(#glow)" />
                      )}
                    </motion.g>
                  </g>
                );
              })}
            </>
          )}

          {/* Heatmap mode — all bounce dots static */}
          {mode === 'heatmap' && (
            <>
              {bounces.map((bounce, idx) => {
                if (!bounce.pos_2d) return null;
                const isOut = bounce.status.includes('Out');
                const col = getHitterColor(bounce.frame);
                return (
                  <motion.circle
                    key={`hmb-${idx}`}
                    cx={px(bounce.pos_2d[0])}
                    cy={py(bounce.pos_2d[1])}
                    r={isOut ? 3 : 4}
                    fill={isOut ? 'none' : col}
                    stroke={col}
                    strokeWidth={isOut ? 1.5 : 0.5}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: isOut ? 0.5 : 0.85 }}
                    transition={{ delay: idx * 0.01, type: 'spring', stiffness: 300, damping: 20 }}
                  />
                );
              })}
            </>
          )}
        </svg>
      </div>

      {/* Legend */}
      <div
        className="mx-5 mb-5 rounded-xl p-3 flex items-center justify-around text-xs"
        style={{ background: 'rgba(255,255,255,0.03)' }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: P1 }} />
          <span style={{ color: 'rgba(255,255,255,0.45)' }}>Player 1</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: P2 }} />
          <span style={{ color: 'rgba(255,255,255,0.45)' }}>Player 2</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full border border-white/30" style={{ background: 'transparent' }} />
          <span style={{ color: 'rgba(255,255,255,0.45)' }}>Out</span>
        </div>
        {mode === 'heatmap' && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: 'rgba(245,80,50,0.7)' }} />
            <span style={{ color: 'rgba(255,255,255,0.45)' }}>Frequency</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default Court2D;
