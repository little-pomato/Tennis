import React from 'react';
import { motion } from 'framer-motion';

interface Props {
  progress: number;
  statusMessage: string;
}

const steps = [
  { label: 'Extracting frames', threshold: 10 },
  { label: 'Initializing AI models', threshold: 30 },
  { label: 'Running inference', threshold: 50 },
  { label: 'Analyzing strokes', threshold: 75 },
  { label: 'Exporting results', threshold: 90 },
];

const ProcessingView: React.FC<Props> = ({ progress, statusMessage }) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-sm text-center space-y-10"
      >
        {/* Orbital spinner */}
        <div className="relative w-28 h-28 mx-auto">
          {/* Outer ring */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-0 rounded-full"
            style={{ border: '2px solid transparent', borderTopColor: '#f5c518', borderRightColor: 'rgba(245,197,24,0.3)' }}
          />
          {/* Inner ring */}
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-4 rounded-full"
            style={{ border: '2px solid transparent', borderTopColor: 'rgba(245,197,24,0.5)', borderLeftColor: 'rgba(245,197,24,0.2)' }}
          />
          {/* Center glow */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-8 h-8 rounded-full"
              style={{ background: 'rgba(245,197,24,0.3)', boxShadow: '0 0 20px rgba(245,197,24,0.5)' }}
            />
          </div>
          {/* Progress arc text */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-base font-bold" style={{ color: '#f5c518' }}>
              {progress}%
            </span>
          </div>
        </div>

        {/* Status text */}
        <div className="space-y-2">
          <motion.h2
            key={statusMessage}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xl font-bold text-white"
          >
            {statusMessage}
          </motion.h2>
          <p className="text-gray-500 text-sm">
            This may take a few minutes depending on video length
          </p>
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <div
            className="w-full h-1.5 rounded-full overflow-hidden"
            style={{ background: 'rgba(255,255,255,0.06)' }}
          >
            <motion.div
              className="h-full rounded-full"
              style={{ background: 'linear-gradient(90deg, #f5c518, #fde047)', boxShadow: '0 0 12px rgba(245,197,24,0.5)' }}
              initial={{ width: '0%' }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            />
          </div>
        </div>

        {/* Step checklist */}
        <div className="space-y-2 text-left">
          {steps.map((step, i) => {
            const done = progress > step.threshold;
            const active = !done && (i === 0 || progress > steps[i - 1].threshold);
            return (
              <motion.div
                key={step.label}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
                className="flex items-center gap-3"
              >
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 transition-all duration-500"
                  style={{
                    background: done
                      ? '#f5c518'
                      : active
                      ? 'rgba(245,197,24,0.2)'
                      : 'rgba(255,255,255,0.05)',
                    border: active ? '1px solid rgba(245,197,24,0.5)' : 'none',
                  }}
                >
                  {done && (
                    <motion.svg
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-3 h-3 text-black"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </motion.svg>
                  )}
                  {active && (
                    <motion.div
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                      className="w-2 h-2 rounded-full"
                      style={{ background: '#f5c518' }}
                    />
                  )}
                </div>
                <span
                  className="text-sm font-medium transition-colors duration-300"
                  style={{ color: done ? 'rgba(255,255,255,0.8)' : active ? '#f5c518' : 'rgba(255,255,255,0.25)' }}
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
