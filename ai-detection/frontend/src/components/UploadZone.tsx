import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  onFile: (file: File) => void;
  onUpload: () => void;
  file: File | null;
}

const TennisBall = () => (
  <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <circle cx="32" cy="32" r="30" fill="#c8e600" />
    <path d="M32 2C17.6 2 5.6 12 2.6 25.4C8.4 25 13.8 22.6 17.4 18.4C21.4 13.8 23 7.4 21.4 2.4C24.8 2.2 28 2 32 2Z" fill="#a8c400" opacity="0.6" />
    <path d="M32 62C46.4 62 58.4 52 61.4 38.6C55.6 39 50.2 41.4 46.6 45.6C42.6 50.2 41 56.6 42.6 61.6C39.2 61.8 36 62 32 62Z" fill="#a8c400" opacity="0.6" />
    <path d="M2 32C2 33.4 2.1 34.8 2.3 36.1C7.2 34.6 11.4 31.4 14 27C16.6 22.6 17 17.4 15 12.8C8.4 18.2 4 24.7 2 32Z" fill="white" opacity="0.3" />
    <path d="M62 32C62 30.6 61.9 29.2 61.7 27.9C56.8 29.4 52.6 32.6 50 37C47.4 41.4 47 46.6 49 51.2C55.6 45.8 60 39.3 62 32Z" fill="white" opacity="0.3" />
  </svg>
);

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
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      {/* Hero text */}
      <motion.div
        initial={{ opacity: 0, y: -24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-16"
      >
        <div className="flex items-center justify-center gap-3 mb-6">
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            className="w-10 h-10"
          >
            <TennisBall />
          </motion.div>
          <span
            className="text-lg font-semibold tracking-widest uppercase"
            style={{ color: '#f5c518', letterSpacing: '0.2em' }}
          >
            TennisAI Analytics
          </span>
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold text-white leading-tight mb-4">
          See Every<br />
          <span style={{ color: '#f5c518' }}>Shot. Every Bounce.</span>
        </h1>
        <p className="text-gray-400 text-xl max-w-lg mx-auto leading-relaxed">
          Upload a match video and get AI-powered ball tracking,
          player movement analysis, and professional-grade stats.
        </p>
      </motion.div>

      {/* Upload zone */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-xl"
      >
        <motion.div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          animate={{ borderColor: isDragging ? 'rgba(245, 197, 24, 0.6)' : 'rgba(255, 255, 255, 0.1)' }}
          className="relative rounded-3xl p-12 text-center cursor-pointer transition-all"
          style={{
            background: isDragging
              ? 'rgba(245, 197, 24, 0.05)'
              : 'rgba(255, 255, 255, 0.02)',
            border: '2px dashed rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(12px)',
          }}
          onClick={() => !file && inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept="video/*"
            onChange={handleChange}
          />

          <AnimatePresence mode="wait">
            {!file ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <motion.div
                  animate={{ y: [0, -8, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                  className="w-20 h-20 mx-auto mb-6"
                >
                  <TennisBall />
                </motion.div>
                <p className="text-xl font-semibold text-white mb-2">
                  Drop your match video here
                </p>
                <p className="text-gray-500 text-sm">
                  MP4, AVI, MOV, MKV · Max 500MB
                </p>
                <div
                  className="mt-6 inline-block px-6 py-2 rounded-full text-sm font-semibold"
                  style={{
                    background: 'rgba(245, 197, 24, 0.15)',
                    color: '#f5c518',
                    border: '1px solid rgba(245, 197, 24, 0.3)',
                  }}
                >
                  Browse files
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="file"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                <motion.div
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center"
                  style={{ background: 'rgba(245, 197, 24, 0.15)' }}
                >
                  <svg className="w-8 h-8" style={{ color: '#f5c518' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </motion.div>
                <div>
                  <p className="font-semibold text-white text-lg">{file.name}</p>
                  <p className="text-gray-500 text-sm mt-1">
                    {(file.size / 1024 / 1024).toFixed(1)} MB · Ready to analyze
                  </p>
                </div>
                <div className="flex gap-3 justify-center">
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={(e) => { e.stopPropagation(); onUpload(); }}
                    className="px-8 py-3 rounded-2xl font-bold text-black text-base shadow-lg"
                    style={{ background: '#f5c518', boxShadow: '0 0 30px rgba(245,197,24,0.3)' }}
                  >
                    Start Analysis
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
                    className="px-5 py-3 rounded-2xl font-semibold text-sm"
                    style={{
                      background: 'rgba(255,255,255,0.06)',
                      color: 'rgba(255,255,255,0.6)',
                      border: '1px solid rgba(255,255,255,0.1)',
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
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
        className="flex flex-wrap justify-center gap-3 mt-10"
      >
        {['Ball Trajectory', 'Player Tracking', 'Bounce Detection', 'Stroke Analysis', '2D Court Map'].map((f) => (
          <span
            key={f}
            className="px-3 py-1.5 rounded-full text-xs font-medium"
            style={{
              background: 'rgba(255,255,255,0.04)',
              color: 'rgba(255,255,255,0.4)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            {f}
          </span>
        ))}
      </motion.div>
    </div>
  );
};

export default UploadZone;
