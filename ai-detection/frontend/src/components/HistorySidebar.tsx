import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { CachedAnalysis } from '../hooks/useAnalysisCache';

interface Props {
  open: boolean;
  onClose: () => void;
  analyses: CachedAnalysis[];
  onLoad: (analysis: CachedAnalysis) => void;
  onRemove: (id: string) => void;
  activeId?: string;
}

function fmt(iso: string) {
  const d = new Date(iso);
  const today = new Date();
  const isToday = d.toDateString() === today.toDateString();
  if (isToday) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

const HistorySidebar: React.FC<Props> = ({ open, onClose, analyses, onLoad, onRemove, activeId }) => {
  return (
    <>
      {/* Backdrop */}
      <AnimatePresence>
        {open && (
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40"
            style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' }}
          />
        )}
      </AnimatePresence>

      {/* Drawer */}
      <AnimatePresence>
        {open && (
          <motion.aside
            key="drawer"
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 380, damping: 38 }}
            className="fixed top-0 left-0 h-full z-50 flex flex-col"
            style={{
              width: 300,
              background: 'rgba(8, 12, 24, 0.98)',
              backdropFilter: 'blur(24px)',
              borderRight: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            {/* Header */}
            <div
              className="flex items-center justify-between px-5 py-4 shrink-0"
              style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
            >
              <div>
                <p className="font-bold text-white text-sm">Match History</p>
                <p className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
                  {analyses.length} saved · local cache
                </p>
              </div>
              <button
                onClick={onClose}
                className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                style={{ background: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.5)' }}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto py-3 px-3 space-y-2">
              {analyses.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full pb-16 gap-3">
                  <div
                    className="w-12 h-12 rounded-2xl flex items-center justify-center"
                    style={{ background: 'rgba(255,255,255,0.04)' }}
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="1.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 7h18M3 12h18M3 17h12" />
                    </svg>
                  </div>
                  <p className="text-xs text-center" style={{ color: 'rgba(255,255,255,0.2)' }}>
                    No analyses yet.<br />Results are cached automatically.
                  </p>
                </div>
              ) : (
                analyses.map((a, i) => {
                  const isActive = a.id === activeId;
                  return (
                    <motion.div
                      key={a.id}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                      className="group rounded-xl p-3 cursor-pointer transition-all relative"
                      style={{
                        background: isActive
                          ? 'rgba(245,197,24,0.08)'
                          : 'rgba(255,255,255,0.03)',
                        border: isActive
                          ? '1px solid rgba(245,197,24,0.25)'
                          : '1px solid rgba(255,255,255,0.05)',
                      }}
                      onClick={() => onLoad(a)}
                    >
                      {/* Active dot */}
                      {isActive && (
                        <div
                          className="absolute top-3 right-3 w-1.5 h-1.5 rounded-full"
                          style={{ background: '#f5c518' }}
                        />
                      )}

                      {/* File name */}
                      <p
                        className="text-sm font-semibold truncate pr-6 leading-tight"
                        style={{ color: isActive ? '#f5c518' : 'rgba(255,255,255,0.8)' }}
                      >
                        {a.filename}
                      </p>

                      {/* Mini stats row */}
                      <div className="flex items-center gap-3 mt-2">
                        <span className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
                          {fmt(a.analyzedAt)}
                        </span>
                        <span className="text-xs" style={{ color: 'rgba(255,255,255,0.2)' }}>·</span>
                        <span className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
                          {a.stats.totalBounces} bounces
                        </span>
                        {a.stats.avgSpeed > 0 && (
                          <>
                            <span className="text-xs" style={{ color: 'rgba(255,255,255,0.2)' }}>·</span>
                            <span className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
                              {a.stats.avgSpeed.toFixed(0)} km/h
                            </span>
                          </>
                        )}
                      </div>

                      {/* Accuracy bar */}
                      {a.stats.totalBounces > 0 && (
                        <div
                          className="mt-2 h-1 rounded-full overflow-hidden"
                          style={{ background: 'rgba(255,255,255,0.06)' }}
                        >
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${(a.stats.inCount / a.stats.totalBounces) * 100}%`,
                              background: isActive ? '#f5c518' : 'rgba(52,211,153,0.5)',
                            }}
                          />
                        </div>
                      )}

                      {/* Delete button - appears on hover */}
                      <button
                        onClick={(e) => { e.stopPropagation(); onRemove(a.id); }}
                        className="absolute bottom-3 right-2.5 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-md"
                        style={{ color: 'rgba(239,68,68,0.6)' }}
                        title="Remove"
                      >
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </motion.div>
                  );
                })
              )}
            </div>

            {/* Footer hint */}
            <div
              className="px-5 py-4 shrink-0"
              style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
            >
              <p className="text-xs leading-relaxed" style={{ color: 'rgba(255,255,255,0.2)' }}>
                Stored locally · Max 10 entries<br />
                Video overlay requires re-selecting the file
              </p>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
};

export default HistorySidebar;
