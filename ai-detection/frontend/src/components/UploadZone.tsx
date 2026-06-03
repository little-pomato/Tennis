import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  onFile: (file: File) => void;
  onUpload: () => void;
  file: File | null;
}

const TennisBall = () => (
  <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <circle cx="32" cy="32" r="30" fill="#C8E600" />
    <circle cx="32" cy="32" r="30" fill="url(#ballShine)" />
    <path d="M32 2C17.6 2 5.6 12 2.6 25.4C8.4 25 13.8 22.6 17.4 18.4C21.4 13.8 23 7.4 21.4 2.4C24.8 2.2 28 2 32 2Z" fill="#A8C000" opacity="0.55" />
    <path d="M32 62C46.4 62 58.4 52 61.4 38.6C55.6 39 50.2 41.4 46.6 45.6C42.6 50.2 41 56.6 42.6 61.6C39.2 61.8 36 62 32 62Z" fill="#A8C000" opacity="0.55" />
    <path d="M2 32C2 33.4 2.1 34.8 2.3 36.1C7.2 34.6 11.4 31.4 14 27C16.6 22.6 17 17.4 15 12.8C8.4 18.2 4 24.7 2 32Z" fill="white" opacity="0.25" />
    <path d="M62 32C62 30.6 61.9 29.2 61.7 27.9C56.8 29.4 52.6 32.6 50 37C47.4 41.4 47 46.6 49 51.2C55.6 45.8 60 39.3 62 32Z" fill="white" opacity="0.25" />
    <defs>
      <radialGradient id="ballShine" cx="35%" cy="30%" r="55%">
        <stop offset="0%" stopColor="white" stopOpacity="0.18" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
    </defs>
  </svg>
);

// Physics bounce: y drops, squash on impact, stretch on rebound, damp each cycle
const BOUNCE_Y         = [0, 72, 0, 44, 0, 22, 0];
const BOUNCE_SCALE_X   = [1, 1.55, 1, 1.32, 1, 1.18, 1];
const BOUNCE_SCALE_Y   = [1, 0.38, 1.25, 0.55, 1.12, 0.72, 1];
const BOUNCE_TIMES     = [0, 0.26, 0.46, 0.66, 0.78, 0.90, 1];
const SHADOW_SCALE_X   = [0.7, 1.9, 0.75, 1.5, 0.85, 1.25, 0.9];
const SHADOW_OPACITY   = [0.10, 0.35, 0.12, 0.26, 0.14, 0.20, 0.12];

const bounceTrans = {
  duration: 2.4,
  times: BOUNCE_TIMES,
  ease: 'easeIn' as const,
  repeat: Infinity,
  repeatDelay: 0.6,
};

const UploadZone: React.FC<Props> = ({ onFile, onUpload, file }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) onFile(f);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) onFile(e.target.files[0]);
  };

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-16"
      style={{ background: 'var(--surface)' }}
    >
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -28 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.75, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-14"
      >
        <div className="flex items-center justify-center gap-3 mb-7">
          {/* Spinning ball */}
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 9, repeat: Infinity, ease: 'linear' }}
            className="w-9 h-9"
          >
            <TennisBall />
          </motion.div>
          <span
            className="text-base font-bold tracking-widest uppercase"
            style={{ color: 'var(--primary)', letterSpacing: '0.22em', fontFamily: "'Space Mono', monospace" }}
          >
            TennisAI Analytics
          </span>
        </div>

        <h1
          className="text-5xl md:text-6xl font-bold leading-tight mb-4"
          style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}
        >
          See Every
          <br />
          <span style={{ color: 'var(--primary)' }}>Shot. Bounce.</span>
        </h1>
        <p className="text-base max-w-md mx-auto leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          Upload a match video and get AI-powered ball tracking,
          player movement, and professional-grade stats.
        </p>
      </motion.div>

      {/* Upload card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.18, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-xl"
      >
        <motion.div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className="relative rounded-3xl p-12 text-center cursor-pointer"
          style={{
            background: 'var(--surface)',
            boxShadow: isDragging
              ? 'inset 6px 6px 14px rgba(0,102,102,0.18), inset -6px -6px 14px #FFFFFF'
              : 'var(--shadow-raised)',
            border: isDragging
              ? '2px solid rgba(0,102,102,0.35)'
              : '2px solid transparent',
            transition: 'box-shadow 0.25s ease, border 0.25s ease',
          }}
          onClick={() => !file && inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" className="hidden" accept="video/*" onChange={handleChange} />

          <AnimatePresence mode="wait">
            {!file ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {/* Bouncing ball with physics */}
                <div className="relative w-24 h-36 mx-auto mb-5 flex flex-col items-center justify-end">
                  {/* Ground shadow */}
                  <motion.div
                    className="absolute bottom-0 rounded-full"
                    style={{
                      width: 48, height: 10,
                      background: 'radial-gradient(ellipse, rgba(30,41,56,0.25) 0%, transparent 70%)',
                    }}
                    animate={{ scaleX: SHADOW_SCALE_X, opacity: SHADOW_OPACITY }}
                    transition={bounceTrans}
                  />
                  {/* Tennis ball */}
                  <motion.div
                    className="w-16 h-16 absolute"
                    style={{ bottom: 12 }}
                    animate={{
                      y: BOUNCE_Y.map((v) => -v),
                      scaleX: BOUNCE_SCALE_X,
                      scaleY: BOUNCE_SCALE_Y,
                    }}
                    transition={bounceTrans}
                  >
                    <TennisBall />
                  </motion.div>
                </div>

                <p
                  className="text-xl font-bold mb-2"
                  style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}
                >
                  Drop your match video here
                </p>
                <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
                  MP4, AVI, MOV, MKV · Max 500 MB
                </p>
                <button
                  className="px-7 py-2.5 rounded-full text-sm font-bold"
                  style={{
                    background: 'var(--surface)',
                    color: 'var(--primary)',
                    boxShadow: 'var(--shadow-raised-sm)',
                    border: '1.5px solid rgba(0,102,102,0.2)',
                    fontFamily: "'Space Mono', monospace",
                  }}
                >
                  Browse files
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="file"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-5"
              >
                {/* Video icon neumorphic disc */}
                <motion.div
                  animate={{ scale: [1, 1.06, 1] }}
                  transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
                  className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center"
                  style={{ boxShadow: 'var(--shadow-raised-sm)', background: 'var(--surface)' }}
                >
                  <svg className="w-7 h-7" style={{ color: 'var(--primary)' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </motion.div>

                <div>
                  <p
                    className="font-bold text-lg"
                    style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}
                  >
                    {file.name}
                  </p>
                  <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                    {(file.size / 1024 / 1024).toFixed(1)} MB · Ready to analyze
                  </p>
                </div>

                <div className="flex gap-3 justify-center">
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97, boxShadow: 'var(--shadow-pressed)' }}
                    onClick={(e) => { e.stopPropagation(); onUpload(); }}
                    className="px-8 py-3 rounded-2xl font-bold text-base"
                    style={{
                      background: 'var(--primary)',
                      color: 'white',
                      boxShadow: '4px 4px 10px rgba(0,102,102,0.4), -2px -2px 6px rgba(0,102,102,0.1)',
                      fontFamily: "'Space Mono', monospace",
                    }}
                  >
                    Start Analysis
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
                    className="px-5 py-3 rounded-2xl font-bold text-sm"
                    style={{
                      background: 'var(--surface)',
                      color: 'var(--text-muted)',
                      boxShadow: 'var(--shadow-raised-sm)',
                      fontFamily: "'Space Mono', monospace",
                    }}
                  >
                    Change
                  </motion.button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>

      {/* Feature pills */}
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
        className="flex flex-wrap justify-center gap-3 mt-10"
      >
        {['Ball Trajectory', 'Player Tracking', 'Bounce Detection', 'Stroke Analysis', '2D Court Map'].map((f, i) => (
          <motion.span
            key={f}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55 + i * 0.06 }}
            className="px-4 py-2 rounded-full text-xs font-bold"
            style={{
              background: 'var(--surface)',
              color: 'var(--text-muted)',
              boxShadow: 'var(--shadow-raised-sm)',
              fontFamily: "'Space Mono', monospace",
            }}
          >
            {f}
          </motion.span>
        ))}
      </motion.div>
    </div>
  );
};

export default UploadZone;
