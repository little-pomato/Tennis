import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ShotHeatmap from './ShotHeatmap';

interface PlayerStats { forehands: number; backhands: number; }
interface Props {
  side: 'top' | 'bottom';
  stats: PlayerStats;
  bounces: any[];
  ballSpeeds: any[];
}

const CONFIG = {
  top: {
    label: 'Player 1', sublabel: 'Top Court',
    color: '#E05240', glowRgb: '224,82,64',
  },
  bottom: {
    label: 'Player 2', sublabel: 'Bottom Court',
    color: '#2B8C8C', glowRgb: '43,140,140',
  },
};

const AnimatedBar: React.FC<{ pct: number; color: string; label: string; count: number }> = ({
  pct, color, label, count,
}) => (
  <div className="space-y-1.5">
    <div className="flex justify-between items-center">
      <span className="text-xs font-bold" style={{ color: 'var(--text-muted)', fontFamily: "'Space Mono', monospace" }}>
        {label}
      </span>
      <span
        className="text-sm font-bold font-mono-nums"
        style={{ color: 'var(--text)', fontFamily: "'JetBrains Mono', monospace" }}
      >
        {count}
      </span>
    </div>
    <div
      className="w-full h-2 rounded-full"
      style={{ boxShadow: 'var(--shadow-pressed-sm)', background: 'var(--surface)' }}
    >
      <motion.div
        className="h-full rounded-full"
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        style={{ background: color, boxShadow: `0 0 8px rgba(${color.replace('#','').match(/.{2}/g)?.map(h=>parseInt(h,16)).join(',')},0.35)` }}
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
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-3xl p-5 flex-1 min-w-0 relative overflow-hidden"
      style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-raised)' }}
    >
      {/* Corner warm glow accent */}
      <div
        className="absolute top-0 left-0 w-44 h-32 pointer-events-none"
        style={{
          background: `radial-gradient(ellipse at top left, rgba(${cfg.glowRgb},0.12), transparent 72%)`,
        }}
      />

      {/* Header */}
      <div className="relative flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm"
            style={{
              background: 'var(--surface)',
              color: cfg.color,
              boxShadow: `3px 3px 7px rgba(${cfg.glowRgb},0.3), -2px -2px 5px #FFFFFF`,
              fontFamily: "'Space Mono', monospace",
            }}
          >
            {side === 'top' ? 'P1' : 'P2'}
          </div>
          <div>
            <p className="font-bold text-sm" style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}>
              {cfg.label}
            </p>
            <p className="text-[11px]" style={{ color: 'var(--text-subtle)' }}>{cfg.sublabel}</p>
          </div>
        </div>

        {/* Tab toggle */}
        <div
          className="flex rounded-xl p-1 gap-0.5"
          style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
        >
          {(['strokes', 'shotmap'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="relative px-3 py-1.5 rounded-lg text-[11px] font-bold capitalize transition-all"
              style={{
                color: tab === t ? 'white' : 'var(--text-muted)',
                fontFamily: "'Space Mono', monospace",
              }}
            >
              {tab === t && (
                <motion.div
                  layoutId={`tab-pill-${side}`}
                  className="absolute inset-0 rounded-lg"
                  style={{ background: cfg.color, boxShadow: `2px 2px 6px rgba(${cfg.glowRgb},0.4)` }}
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
                {/* Big stat pair — neumorphic pressed insets */}
                <div className="grid grid-cols-2 gap-3">
                  <div
                    className="rounded-2xl p-3 text-center"
                    style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
                  >
                    <p
                      className="text-2xl font-extrabold font-mono-nums"
                      style={{ color: cfg.color, fontFamily: "'JetBrains Mono', monospace" }}
                    >
                      {stats.forehands}
                    </p>
                    <p className="text-[10px] font-bold mt-0.5" style={{ color: 'var(--text-subtle)', fontFamily: "'Space Mono', monospace" }}>
                      Forehand
                    </p>
                  </div>
                  <div
                    className="rounded-2xl p-3 text-center"
                    style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
                  >
                    <p
                      className="text-2xl font-extrabold font-mono-nums"
                      style={{ color: 'var(--text)', fontFamily: "'JetBrains Mono', monospace" }}
                    >
                      {stats.backhands}
                    </p>
                    <p className="text-[10px] font-bold mt-0.5" style={{ color: 'var(--text-subtle)', fontFamily: "'Space Mono', monospace" }}>
                      Backhand
                    </p>
                  </div>
                </div>

                {/* Animated bars */}
                <div className="space-y-2.5">
                  <AnimatedBar pct={fhPct} color={cfg.color} label="Forehand" count={stats.forehands} />
                  <AnimatedBar pct={bhPct} color={`${cfg.color}88`} label="Backhand" count={stats.backhands} />
                </div>

                {/* Ratio strip */}
                <div
                  className="flex items-center gap-2.5 rounded-2xl px-3 py-2.5"
                  style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
                >
                  <div className="flex-1 h-2 rounded-full overflow-hidden"
                    style={{ background: 'var(--surface-dark)' }}>
                    <motion.div
                      className="h-full rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${fhPct}%` }}
                      transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
                      style={{ background: cfg.color }}
                    />
                  </div>
                  <span
                    className="text-[10px] font-bold shrink-0"
                    style={{ color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {fhPct}% FH
                  </span>
                </div>
              </>
            ) : (
              <div
                className="rounded-2xl p-6 text-center text-sm"
                style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)', color: 'var(--text-subtle)' }}
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
            <div
              className="shrink-0 rounded-xl overflow-hidden"
              style={{ boxShadow: 'var(--shadow-pressed-sm)' }}
            >
              <ShotHeatmap
                bounces={bounces}
                ballSpeeds={ballSpeeds}
                playerSide={side}
                playerColor={cfg.color}
              />
            </div>

            {/* Legend + stats */}
            <div className="flex-1 min-w-0 space-y-3 pt-1">
              <div>
                <p className="text-xs font-bold" style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}>
                  Shot Placement
                </p>
                <p className="text-[11px] leading-relaxed mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  Hotter zones = more shots landed there
                </p>
              </div>

              {/* Density gradient */}
              <div className="space-y-1.5">
                <p
                  className="text-[10px] font-bold uppercase tracking-wider"
                  style={{ color: 'var(--text-subtle)', fontFamily: "'Space Mono', monospace" }}
                >
                  Density
                </p>
                <div
                  className="h-2 w-full rounded-full"
                  style={{ background: `linear-gradient(90deg, transparent, ${cfg.color}66, ${cfg.color})` }}
                />
                <div className="flex justify-between">
                  <span className="text-[9px]" style={{ color: 'var(--text-subtle)' }}>Low</span>
                  <span className="text-[9px]" style={{ color: 'var(--text-subtle)' }}>High</span>
                </div>
              </div>

              {/* Shot count */}
              <div
                className="rounded-xl p-2.5 text-center"
                style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
              >
                <p
                  className="text-xl font-extrabold font-mono-nums"
                  style={{ color: cfg.color, fontFamily: "'JetBrains Mono', monospace" }}
                >
                  {bounces.filter((b) => {
                    if (!b.pos_2d) return false;
                    const sorted = [...ballSpeeds].sort((a, c) => a.end - c.end);
                    const exact = sorted.find((s) => s.end === b.frame);
                    if (exact) return exact.side === side;
                    const isTopHalf = b.pos_2d[1] < 11.885;
                    return side === 'top' ? !isTopHalf : isTopHalf;
                  }).length}
                </p>
                <p className="text-[10px]" style={{ color: 'var(--text-subtle)' }}>shots tracked</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default PlayerCard;
