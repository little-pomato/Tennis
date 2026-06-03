import React from 'react';
import { motion } from 'framer-motion';

interface Props {
  progress: number;
  statusMessage: string;
}

const steps = [
  { label: 'Extracting frames',     threshold: 10 },
  { label: 'Initializing AI models',threshold: 30 },
  { label: 'Running inference',     threshold: 50 },
  { label: 'Analyzing strokes',     threshold: 75 },
  { label: 'Exporting results',     threshold: 90 },
];

const ProcessingView: React.FC<Props> = ({ progress, statusMessage }) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4" style={{ background: 'var(--surface)' }}>
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-sm text-center space-y-10"
      >
        {/* Spinner disc */}
        <div className="relative w-32 h-32 mx-auto">
          {/* Neumorphic base disc */}
          <div
            className="absolute inset-0 rounded-full"
            style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-raised)' }}
          />

          {/* Outer arc */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-2 rounded-full"
            style={{
              border: '3px solid transparent',
              borderTopColor: 'var(--primary)',
              borderRightColor: 'rgba(0,102,102,0.25)',
            }}
          />

          {/* Inner counter-arc */}
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 1.9, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-6 rounded-full"
            style={{
              border: '2.5px solid transparent',
              borderTopColor: 'var(--warning)',
              borderLeftColor: 'rgba(254,153,0,0.25)',
            }}
          />

          {/* Center pulsing glow */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              animate={{ scale: [1, 1.25, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              className="w-9 h-9 rounded-full"
              style={{ background: 'rgba(0,102,102,0.18)', boxShadow: '0 0 18px rgba(0,102,102,0.35)' }}
            />
          </div>

          {/* Progress % */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span
              className="text-base font-bold font-mono-nums"
              style={{ color: 'var(--primary)', fontFamily: "'JetBrains Mono', monospace" }}
            >
              {progress}%
            </span>
          </div>
        </div>

        {/* Status */}
        <div className="space-y-2">
          <motion.h2
            key={statusMessage}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xl font-bold"
            style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}
          >
            {statusMessage}
          </motion.h2>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            This may take a few minutes depending on video length
          </p>
        </div>

        {/* Progress bar */}
        <div
          className="w-full h-2.5 rounded-full overflow-hidden"
          style={{ boxShadow: 'var(--shadow-pressed-sm)', background: 'var(--surface)' }}
        >
          <motion.div
            className="h-full rounded-full"
            style={{
              background: 'linear-gradient(90deg, var(--primary), #009999, var(--warning))',
              boxShadow: '0 0 12px rgba(0,102,102,0.45)',
            }}
            initial={{ width: '0%' }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>

        {/* Steps */}
        <div className="space-y-3 text-left">
          {steps.map((step, i) => {
            const done   = progress > step.threshold;
            const active = !done && (i === 0 || progress > steps[i - 1].threshold);
            return (
              <motion.div
                key={step.label}
                initial={{ opacity: 0, x: -14 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.09 }}
                className="flex items-center gap-3"
              >
                {/* Step indicator */}
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 transition-all duration-500"
                  style={{
                    background: done   ? 'var(--primary)' : 'var(--surface)',
                    boxShadow: done
                      ? '2px 2px 6px rgba(0,102,102,0.4)'
                      : active
                      ? 'var(--shadow-pressed-sm)'
                      : 'var(--shadow-raised-sm)',
                  }}
                >
                  {done && (
                    <motion.svg
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-3.5 h-3.5"
                      fill="none" viewBox="0 0 24 24" stroke="white" strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </motion.svg>
                  )}
                  {active && (
                    <motion.div
                      animate={{ scale: [1, 1.35, 1], opacity: [0.7, 1, 0.7] }}
                      transition={{ duration: 1, repeat: Infinity }}
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ background: 'var(--warning)' }}
                    />
                  )}
                </div>

                <span
                  className="text-sm font-bold transition-colors duration-300"
                  style={{
                    color: done   ? 'var(--primary)'
                         : active ? 'var(--warning)'
                         :          'var(--text-subtle)',
                    fontFamily: "'Space Mono', monospace",
                  }}
                >
                  {step.label}
                </span>
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
};

export default ProcessingView;
