import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ShotHeatmap from './ShotHeatmap';

interface PlayerStats {
  forehands: number;
  backhands: number;
}

interface Props {
  side: 'top' | 'bottom';
  stats: PlayerStats;
  bounces: any[];
  ballSpeeds: any[];
}

const CONFIG = {
  top: {
    label: 'Player 1',
    sublabel: 'Top Court',
    color: '#ff6060',
    hex: '#ff6060',
    glow: 'rgba(255,96,96,0.2)',
    borderColor: 'rgba(255,96,96,0.2)',
  },
  bottom: {
    label: 'Player 2',
    sublabel: 'Bottom Court',
    color: '#4d9eff',
    hex: '#4d9eff',
    glow: 'rgba(77,158,255,0.2)',
    borderColor: 'rgba(77,158,255,0.2)',
  },
};

const AnimatedBar: React.FC<{ pct: number; color: string; label: string; count: number }> = ({
  pct, color, label, count,
}) => (
  <div className="space-y-1.5">
    <div className="flex justify-between items-center">
      <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.45)' }}>{label}</span>
      <span className="text-sm font-bold text-white">{count}</span>
    </div>
    <div className="w-full h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
      <motion.div
        className="h-full rounded-full"
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        style={{ background: color }}
      />
    </div>
  </div>
);

type Tab = 'strokes' | 'shotmap';

const PlayerCard: React.FC<Props> = ({ side, stats, bounces, ballSpeeds }) => {
  const cfg = CONFIG[side];
  const [tab, setTab] = useState<Tab>('strokes');
  const total = stats.forehands + stats.backhands;
  const fhPct = total > 0 ? Math.round((stats.forehands / total) * 100) : 0;
  const bhPct = total > 0 ? 100 - fhPct : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-2xl p-5 flex-1 min-w-0 relative overflow-hidden"
      style={{
        background: 'rgba(255,255,255,0.02)',
        border: `1px solid ${cfg.borderColor}`,
        backdropFilter: 'blur(12px)',
      }}
    >
      {/* Corner glow */}
      <div
        className="absolute top-0 left-0 w-40 h-28 pointer-events-none"
        style={{
          background: `radial-gradient(ellipse at top left, ${cfg.glow}, transparent 70%)`,
        }}
      />

      {/* Header */}
      <div className="relative flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm"
            style={{ background: `${cfg.color}1a`, color: cfg.color, border: `1px solid ${cfg.color}33` }}
          >
            {side === 'top' ? 'P1' : 'P2'}
          </div>
          <div>
            <p className="font-bold text-white text-sm">{cfg.label}</p>
            <p className="text-[11px]" style={{ color: 'rgba(255,255,255,0.3)' }}>{cfg.sublabel}</p>
          </div>
        </div>

        {/* Tab toggle */}
        <div
          className="flex rounded-xl p-0.5 gap-0.5"
          style={{ background: 'rgba(255,255,255,0.05)' }}
        >
          {(['strokes', 'shotmap'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="relative px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors capitalize"
              style={{ color: tab === t ? 'black' : 'rgba(255,255,255,0.35)' }}
            >
              {tab === t && (
                <motion.div
                  layoutId={`tab-pill-${side}`}
                  className="absolute inset-0 rounded-lg"
                  style={{ background: cfg.color }}
                  transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                />
              )}
              <span className="relative">{t === 'shotmap' ? 'Shot Map' : 'Strokes'}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <AnimatePresence mode="wait">
        {tab === 'strokes' ? (
          <motion.div
            key="strokes"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
            transition={{ duration: 0.25 }}
            className="space-y-4"
          >
            {total > 0 ? (
              <>
                {/* Big stat pair */}
                <div className="grid grid-cols-2 gap-2">
                  <div
                    className="rounded-xl p-3 text-center"
                    style={{ background: `${cfg.color}11` }}
                  >
                    <p className="text-2xl font-extrabold" style={{ color: cfg.color }}>{stats.forehands}</p>
                    <p className="text-[10px] font-medium mt-0.5" style={{ color: 'rgba(255,255,255,0.35)' }}>Forehand</p>
                  </div>
                  <div
                    className="rounded-xl p-3 text-center"
                    style={{ background: 'rgba(255,255,255,0.04)' }}
                  >
                    <p className="text-2xl font-extrabold text-white">{stats.backhands}</p>
                    <p className="text-[10px] font-medium mt-0.5" style={{ color: 'rgba(255,255,255,0.35)' }}>Backhand</p>
                  </div>
                </div>

                {/* Bars */}
                <div className="space-y-2">
                  <AnimatedBar pct={fhPct} color={cfg.color} label="Forehand" count={stats.forehands} />
                  <AnimatedBar pct={bhPct} color={`${cfg.color}77`} label="Backhand" count={stats.backhands} />
                </div>

                {/* Ratio pill */}
                <div
                  className="flex items-center gap-2 rounded-xl px-3 py-2"
                  style={{ background: 'rgba(255,255,255,0.03)' }}
                >
                  <div
                    className="flex-1 h-2 rounded-full overflow-hidden"
                    style={{ background: 'rgba(255,255,255,0.06)' }}
                  >
                    <motion.div
                      className="h-full rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${fhPct}%` }}
                      transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
                      style={{ background: cfg.color }}
                    />
                  </div>
                  <span className="text-[10px] font-semibold shrink-0" style={{ color: 'rgba(255,255,255,0.35)' }}>
                    {fhPct}% FH
                  </span>
                </div>
              </>
            ) : (
              <div
                className="rounded-xl p-6 text-center text-sm"
                style={{ background: 'rgba(255,255,255,0.02)', color: 'rgba(255,255,255,0.2)' }}
              >
                Stroke data unavailable<br />
                <span className="text-[11px]">Requires updated pipeline</span>
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="shotmap"
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.25 }}
            className="flex gap-4 items-start"
          >
            {/* Heatmap canvas */}
            <div className="shrink-0">
              <ShotHeatmap
                bounces={bounces}
                ballSpeeds={ballSpeeds}
                playerSide={side}
                playerColor={cfg.hex}
              />
            </div>

            {/* Legend + stats */}
            <div className="flex-1 min-w-0 space-y-3 pt-1">
              <div>
                <p className="text-xs font-semibold text-white mb-1">Shot Placement</p>
                <p className="text-[11px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.3)' }}>
                  Hotter zones = more shots landed there
                </p>
              </div>

              {/* Color scale */}
              <div className="space-y-1.5">
                <p className="text-[10px] font-medium uppercase tracking-wider" style={{ color: 'rgba(255,255,255,0.25)' }}>Density</p>
                <div
                  className="h-2 w-full rounded-full"
                  style={{
                    background: `linear-gradient(90deg, transparent, ${cfg.color}66, ${cfg.color})`,
                  }}
                />
                <div className="flex justify-between">
                  <span className="text-[9px]" style={{ color: 'rgba(255,255,255,0.2)' }}>Low</span>
                  <span className="text-[9px]" style={{ color: 'rgba(255,255,255,0.2)' }}>High</span>
                </div>
              </div>

              {/* Bounce count for player */}
              <div
                className="rounded-xl p-2.5 text-center"
                style={{ background: `${cfg.color}11` }}
              >
                <p className="text-lg font-extrabold" style={{ color: cfg.color }}>
                  {bounces.filter((b) => {
                    if (!b.pos_2d) return false;
                    const sorted = [...ballSpeeds].sort((a, c) => a.end - c.end);
                    const exact = sorted.find((s) => s.end === b.frame);
                    if (exact) return exact.side === side;
                    // Physics fallback: P1 shots land on bottom half, P2 on top half
                    const isTopHalf = b.pos_2d[1] < 11.885;
                    return side === 'top' ? !isTopHalf : isTopHalf;
                  }).length}
                </p>
                <p className="text-[10px]" style={{ color: 'rgba(255,255,255,0.3)' }}>shots tracked</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default PlayerCard;
