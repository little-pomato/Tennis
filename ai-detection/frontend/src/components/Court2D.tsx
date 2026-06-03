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

// Warm player colors
const P1 = '#E05240';
const P2 = '#2B8C8C';

// Clay court palette
const CLAY_SURFACE  = '#C07940';
const CLAY_BORDER   = '#9A5E28';
const CLAY_LINE     = 'rgba(255,248,235,0.92)';

// Grid for heatmap
const GRID_COLS = 8;
const GRID_ROWS = 14;

// 8-directional dust vectors
const DUST_ANGLES = Array.from({ length: 8 }, (_, i) => (i / 8) * Math.PI * 2);

const Court2D: React.FC<Props> = ({ data, currentFrame }) => {
  const [mode, setMode] = useState<ViewMode>('live');
  const playerTracks = data.results.player_metric_tracks || { top: [], bottom: [] };
  const bounces: any[] = data.results.bounces || [];

  const ballSpeeds: any[] = useMemo(
    () => [...(data.results.analytics?.ball_speeds ?? [])].sort((a, b) => a.end - b.end),
    [data]
  );

  const getHitterColor = (bounceFrame: number): string => {
    const exact = ballSpeeds.find((s) => s.end === bounceFrame);
    if (exact) return exact.side === 'top' ? P1 : P2;
    const next = ballSpeeds.find((s) => s.end >= bounceFrame);
    if (next) return next.side === 'top' ? P1 : P2;
    if (ballSpeeds.length > 0)
      return ballSpeeds[ballSpeeds.length - 1].side === 'top' ? P1 : P2;
    return P1;
  };

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

  // Ball stroke arcs between consecutive bounces
  const strokeArcs = useMemo(() => {
    const arcs: { x1: number; y1: number; x2: number; y2: number; cpX: number; cpY: number; color: string; endFrame: number }[] = [];
    for (let i = 1; i < bounces.length; i++) {
      const prev = bounces[i - 1];
      const curr = bounces[i];
      if (!prev.pos_2d || !curr.pos_2d) continue;
      const x1 = px(prev.pos_2d[0]);
      const y1 = py(prev.pos_2d[1]);
      const x2 = px(curr.pos_2d[0]);
      const y2 = py(curr.pos_2d[1]);
      // Lift control point above the midpoint (ball arcs over the net)
      const cpX = (x1 + x2) / 2;
      const cpY = Math.min(y1, y2) - 22;
      arcs.push({ x1, y1, x2, y2, cpX, cpY, color: getHitterColor(curr.frame), endFrame: curr.frame });
    }
    return arcs;
  }, [bounces, ballSpeeds]);

  return (
    <div
      className="rounded-3xl overflow-hidden"
      style={{
        background: 'var(--surface)',
        boxShadow: 'var(--shadow-raised)',
      }}
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-4 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-sm" style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}>
            Court Map
          </h3>
          <p className="text-xs mt-0.5 font-mono-nums" style={{ color: 'var(--text-muted)' }}>
            {inCount} In · {outCount} Out
          </p>
        </div>
        {/* Mode toggle — neumorphic pill */}
        <div
          className="flex rounded-xl p-1 gap-0.5"
          style={{ boxShadow: 'var(--shadow-pressed-sm)', background: 'var(--surface)' }}
        >
          {(['live', 'heatmap'] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className="relative px-3 py-1.5 rounded-lg text-xs font-bold transition-all capitalize"
              style={{
                color: mode === m ? 'white' : 'var(--text-muted)',
                background: mode === m ? 'var(--primary)' : 'transparent',
                boxShadow: mode === m ? '2px 2px 5px rgba(0,102,102,0.35)' : 'none',
              }}
            >
              {mode === m && (
                <motion.div
                  layoutId="modePill"
                  className="absolute inset-0 rounded-lg"
                  style={{ background: 'var(--primary)', zIndex: -1 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <span className="relative">{m}</span>
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
          className="rounded-2xl"
          style={{ maxWidth: '100%' }}
        >
          <defs>
            {/* Warm glow filter */}
            <filter id="warmGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* P1 glow */}
            <filter id="glowP1" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feFlood floodColor="#E05240" floodOpacity="0.6" result="color" />
              <feComposite in="color" in2="blur" operator="in" result="shadow" />
              <feMerge><feMergeNode in="shadow" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>

            {/* P2 glow */}
            <filter id="glowP2" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feFlood floodColor="#2B8C8C" floodOpacity="0.6" result="color" />
              <feComposite in="color" in2="blur" operator="in" result="shadow" />
              <feMerge><feMergeNode in="shadow" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* Clay border area */}
          <rect x={0} y={0} width={CW} height={CH} fill={CLAY_BORDER} rx={16} />

          {/* Court surface */}
          <rect x={PAD_X} y={PAD_Y} width={COURT.width * SCALE} height={COURT.length * SCALE} fill={CLAY_SURFACE} />

          {/* Subtle clay texture overlay */}
          <rect
            x={PAD_X} y={PAD_Y}
            width={COURT.width * SCALE} height={COURT.length * SCALE}
            fill="url(#clayTexture)"
            opacity={0.08}
          />

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
                        animate={{ opacity: intensity * 0.82 }}
                        exit={{ opacity: 0 }}
                        fill="rgba(255,120,50,1)"
                      />
                    );
                  })
                )}
              </g>
            )}
          </AnimatePresence>

          {/* Court lines */}
          <rect x={px(0)} y={py(0)} width={COURT.width * SCALE} height={COURT.length * SCALE}
            fill="none" stroke={CLAY_LINE} strokeWidth="2.5" />
          <line x1={px(MARGIN)} y1={py(0)} x2={px(MARGIN)} y2={py(COURT.length)} stroke={CLAY_LINE} strokeWidth="1.2" />
          <line x1={px(COURT.width - MARGIN)} y1={py(0)} x2={px(COURT.width - MARGIN)} y2={py(COURT.length)} stroke={CLAY_LINE} strokeWidth="1.2" />
          <line x1={px(MARGIN)} y1={py(COURT.length / 2 - COURT.service_line)} x2={px(COURT.width - MARGIN)} y2={py(COURT.length / 2 - COURT.service_line)} stroke={CLAY_LINE} strokeWidth="1.2" />
          <line x1={px(MARGIN)} y1={py(COURT.length / 2 + COURT.service_line)} x2={px(COURT.width - MARGIN)} y2={py(COURT.length / 2 + COURT.service_line)} stroke={CLAY_LINE} strokeWidth="1.2" />
          <line x1={px(COURT.width / 2)} y1={py(COURT.length / 2 - COURT.service_line)} x2={px(COURT.width / 2)} y2={py(COURT.length / 2 + COURT.service_line)} stroke={CLAY_LINE} strokeWidth="1.2" />

          {/* Net — thick dark band with white dashes */}
          <line x1={px(-0.4)} y1={py(COURT.length / 2)} x2={px(COURT.width + 0.4)} y2={py(COURT.length / 2)}
            stroke="rgba(30,20,10,0.85)" strokeWidth="5" />
          <line x1={px(-0.4)} y1={py(COURT.length / 2)} x2={px(COURT.width + 0.4)} y2={py(COURT.length / 2)}
            stroke="rgba(255,248,235,0.45)" strokeWidth="1.2" strokeDasharray="5,4" />

          {/* Player zone labels */}
          <text x={px(COURT.width / 2)} y={py(COURT.length * 0.10)} textAnchor="middle" fill={P1} fontSize="9" fontWeight="700" opacity="0.7" fontFamily="Space Mono">P1</text>
          <text x={px(COURT.width / 2)} y={py(COURT.length * 0.92)} textAnchor="middle" fill={P2} fontSize="9" fontWeight="700" opacity="0.7" fontFamily="Space Mono">P2</text>

          {/* Live mode */}
          {mode === 'live' && (
            <>
              {/* Ball stroke arcs — trajectory between bounces */}
              {strokeArcs.map((arc, i) => {
                const age = currentFrame - arc.endFrame;
                if (age < 0 || age > 90) return null;
                const opacity = Math.max(0, 0.55 * (1 - age / 90));
                return (
                  <path
                    key={`arc-${i}`}
                    d={`M ${arc.x1} ${arc.y1} Q ${arc.cpX} ${arc.cpY} ${arc.x2} ${arc.y2}`}
                    fill="none"
                    stroke={arc.color}
                    strokeWidth={1.8}
                    strokeOpacity={opacity}
                    strokeDasharray="5 3.5"
                    strokeLinecap="round"
                  />
                );
              })}

              {/* Player trails */}
              {(['top', 'bottom'] as const).map((side) => {
                const track = playerTracks[side];
                const window = 3;
                const trailLength = 45;
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
                    const t = 1 - age / trailLength;
                    const opacity = Math.max(0, 0.55 * t);
                    segments.push(
                      <line
                        key={`${side}-${i}`}
                        x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                        stroke={side === 'top' ? P1 : P2}
                        strokeWidth={3 * t}
                        strokeOpacity={opacity}
                        strokeLinecap="round"
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
                const glowId = side === 'top' ? 'glowP1' : 'glowP2';
                return (
                  <g key={`dot-${side}`}>
                    <motion.circle
                      cx={px(pt.x)} cy={py(pt.y)}
                      r="14"
                      fill={color}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: [0.06, 0.18, 0.06], scale: [1, 1.6, 1] }}
                      transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
                    />
                    <motion.circle
                      r="7"
                      fill={color}
                      stroke="rgba(255,248,235,0.9)"
                      strokeWidth="2.5"
                      filter={`url(#${glowId})`}
                      animate={{ cx: px(pt.x), cy: py(pt.y) }}
                      transition={{ type: 'spring', damping: 20, stiffness: 150 }}
                    />
                  </g>
                );
              })}

              {/* Bounce dots with squash/stretch + shockwave */}
              {bounces.map((bounce, idx) => {
                if (!bounce.pos_2d) return null;
                const diff = currentFrame - bounce.frame;
                if (diff < 0 || diff > 70) return null;

                const isFresh     = diff < 8;
                const isShockwave = diff < 38;   // rings play for ~1.25 s
                const isNear      = diff < 30;
                const isOut       = bounce.status.includes('Out');
                const col         = getHitterColor(bounce.frame);
                const bx          = px(bounce.pos_2d[0]);
                const by          = py(bounce.pos_2d[1]);

                return (
                  <g key={`bounce-${idx}`}>
                    {/* Three-layer amber shockwave */}
                    {isShockwave && (
                      <>
                        <motion.circle
                          cx={bx} cy={by}
                          initial={{ r: 2, opacity: 0.92, strokeWidth: 3 }}
                          animate={{ r: 30, opacity: 0, strokeWidth: 0 }}
                          transition={{ duration: 0.55, ease: 'easeOut' }}
                          fill="none" stroke="#FE9900"
                        />
                        <motion.circle
                          cx={bx} cy={by}
                          initial={{ r: 2, opacity: 0.6, strokeWidth: 2 }}
                          animate={{ r: 52, opacity: 0, strokeWidth: 0 }}
                          transition={{ duration: 0.85, ease: 'easeOut', delay: 0.07 }}
                          fill="none" stroke="#FE9900"
                        />
                        <motion.circle
                          cx={bx} cy={by}
                          initial={{ r: 2, opacity: 0.32, strokeWidth: 1 }}
                          animate={{ r: 78, opacity: 0, strokeWidth: 0 }}
                          transition={{ duration: 1.3, ease: 'easeOut', delay: 0.14 }}
                          fill="none" stroke={CLAY_BORDER}
                        />
                      </>
                    )}

                    {/* Dust particles on impact */}
                    {isFresh && DUST_ANGLES.map((angle, i) => {
                      const dist = 15 + (i % 2) * 6;
                      return (
                        <motion.circle
                          key={`dust-${idx}-${i}`}
                          r={1.8}
                          fill={CLAY_BORDER}
                          initial={{ cx: bx, cy: by, opacity: 0.85 }}
                          animate={{
                            cx: bx + Math.cos(angle) * dist,
                            cy: by + Math.sin(angle) * dist,
                            opacity: 0,
                          }}
                          transition={{ duration: 0.45, ease: 'easeOut', delay: i * 0.015 }}
                        />
                      );
                    })}

                    {/* Bounce dot — squash/stretch on fresh impact */}
                    {isFresh ? (
                      <motion.circle
                        cx={bx} cy={by}
                        r={isOut ? 6 : 6}
                        fill={isOut ? 'none' : col}
                        stroke={isOut ? col : 'rgba(255,248,235,0.9)'}
                        strokeWidth={isOut ? 2 : 1.8}
                        filter="url(#warmGlow)"
                        style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                        initial={{ scaleX: 0, scaleY: 0, opacity: 0 }}
                        animate={{
                          scaleX: [0, 2.1, 0.65, 1.12, 0.95, 1],
                          scaleY: [0, 0.28, 1.6,  0.85, 1.05, 1],
                          opacity: [0, 1,   1,    1,    1,    1],
                        }}
                        transition={{
                          duration: 0.75,
                          times: [0, 0.18, 0.45, 0.65, 0.82, 1],
                          ease: 'easeOut',
                        }}
                      />
                    ) : isOut ? (
                      <g filter="url(#warmGlow)" opacity={isNear ? 1 : 0.4}>
                        <circle cx={bx} cy={by} r={isNear ? 6 : 4} fill="none" stroke={col} strokeWidth="2" />
                        <line x1={bx - 4} y1={by - 4} x2={bx + 4} y2={by + 4} stroke={col} strokeWidth="2" />
                        <line x1={bx + 4} y1={by - 4} x2={bx - 4} y2={by + 4} stroke={col} strokeWidth="2" />
                      </g>
                    ) : (
                      <circle
                        cx={bx} cy={by}
                        r={isNear ? 6 : 3.5}
                        fill={col}
                        stroke="rgba(255,248,235,0.85)"
                        strokeWidth="1.5"
                        opacity={isNear ? 1 : 0.45}
                        filter="url(#warmGlow)"
                      />
                    )}
                  </g>
                );
              })}
            </>
          )}

          {/* Heatmap mode — static bounce dots */}
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
                    r={isOut ? 3.5 : 5}
                    fill={isOut ? 'none' : col}
                    stroke={col}
                    strokeWidth={isOut ? 1.5 : 0.5}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: isOut ? 0.55 : 0.88 }}
                    transition={{ delay: idx * 0.01, type: 'spring', stiffness: 280, damping: 22 }}
                    style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                  />
                );
              })}
            </>
          )}
        </svg>
      </div>

      {/* Legend */}
      <div
        className="mx-5 mb-5 rounded-2xl px-4 py-3 flex items-center justify-around text-xs"
        style={{ boxShadow: 'var(--shadow-pressed-sm)', background: 'var(--surface)' }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: P1 }} />
          <span style={{ color: 'var(--text-muted)' }}>Player 1</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: P2 }} />
          <span style={{ color: 'var(--text-muted)' }}>Player 2</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full border-2" style={{ borderColor: 'var(--text-subtle)', background: 'transparent' }} />
          <span style={{ color: 'var(--text-muted)' }}>Out</span>
        </div>
        {mode === 'heatmap' && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: 'rgba(255,120,50,0.7)' }} />
            <span style={{ color: 'var(--text-muted)' }}>Density</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default Court2D;
