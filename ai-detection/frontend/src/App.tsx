import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import UploadZone from './components/UploadZone';
import ProcessingView from './components/ProcessingView';
import VideoOverlay from './components/VideoOverlay';
import Court2D from './components/Court2D';
import PlayerCard from './components/PlayerCard';
import SpeedChart from './components/SpeedChart';
import HistorySidebar from './components/HistorySidebar';
import { useAnalysisCache } from './hooks/useAnalysisCache';

const P1 = '#ff6060';
const P2 = '#4d9eff';

function StatPill({
  label, value, sub, delay = 0,
}: { label: string; value: string | number; sub?: string; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-2xl px-5 py-4 flex-1 min-w-0"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <p className="text-xs font-medium uppercase tracking-wider mb-1" style={{ color: 'rgba(255,255,255,0.3)' }}>
        {label}
      </p>
      <p className="text-2xl font-extrabold text-white leading-none">{value}</p>
      {sub && <p className="text-xs mt-1" style={{ color: 'rgba(255,255,255,0.22)' }}>{sub}</p>}
    </motion.div>
  );
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeId, setActiveId] = useState<string | undefined>();
  const [needsVideoFile, setNeedsVideoFile] = useState(false);

  const { analyses, saveAnalysis, removeAnalysis } = useAnalysisCache();

  const handleFile = (f: File) => {
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setData(null);
    setError(null);
    setRequestId(null);
    setProgress(0);
    setNeedsVideoFile(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProgress(0);
    setStatusMessage('Uploading video...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setRequestId(res.data.request_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start upload. Ensure backend is running.');
      setLoading(false);
    }
  };

  // Poll for backend status
  useEffect(() => {
    let interval: any;
    if (requestId && loading) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`http://localhost:8000/status/${requestId}`);
          const { status, progress, message, result, error } = res.data;
          setProgress(progress);
          setStatusMessage(message || status);
          if (status === 'completed') {
            setData(result);
            setLoading(false);
            setRequestId(null);
            clearInterval(interval);
            // Auto-cache the result
            if (file) {
              const id = saveAnalysis(file.name, result);
              setActiveId(id);
            }
          } else if (status === 'failed') {
            setError(error || 'Processing failed.');
            setLoading(false);
            setRequestId(null);
            clearInterval(interval);
          }
        } catch { /* ignore poll errors */ }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [requestId, loading]);

  // Load a cached analysis
  const loadFromCache = (analysis: typeof analyses[0]) => {
    setData(analysis.data);
    setActiveId(analysis.id);
    setFile(null);
    setVideoUrl(null);
    setNeedsVideoFile(true);
    setError(null);
    setSidebarOpen(false);
  };

  // Attach video file to a cached analysis
  const attachVideo = (f: File) => {
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setNeedsVideoFile(false);
  };

  const reset = () => {
    setData(null);
    setFile(null);
    setVideoUrl(null);
    setProgress(0);
    setError(null);
    setLoading(false);
    setActiveId(undefined);
    setNeedsVideoFile(false);
  };

  // Derived stats
  const analytics = data?.results?.analytics;
  const bounces: any[] = data?.results?.bounces ?? [];
  const ballSpeeds: any[] = analytics?.ball_speeds ?? [];
  const inCount = bounces.filter((b) => b.status === 'In').length;
  const outCount = bounces.filter((b) => b.status.includes('Out')).length;
  const speedData = ballSpeeds.map((s: any) => ({
    frame: s.frame ?? s.start,
    speed: s.speed_kmh,
  }));
  const avgSpeed = speedData.length > 0
    ? speedData.reduce((a: number, b: any) => a + b.speed, 0) / speedData.length
    : 0;
  const maxSpeed = speedData.length > 0 ? Math.max(...speedData.map((d: any) => d.speed)) : 0;
  const topStats = analytics?.player_stats?.top ?? { forehands: 0, backhands: 0 };
  const bottomStats = analytics?.player_stats?.bottom ?? { forehands: 0, backhands: 0 };

  return (
    <div className="min-h-screen" style={{ background: '#060a14' }}>
      {/* Nav */}
      <nav
        className="sticky top-0 z-30 border-b"
        style={{
          background: 'rgba(6,10,20,0.88)',
          backdropFilter: 'blur(20px)',
          borderColor: 'rgba(255,255,255,0.06)',
        }}
      >
        <div className="max-w-7xl mx-auto px-5 h-14 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            {/* History button */}
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => setSidebarOpen(true)}
              className="relative flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold transition-colors"
              style={{
                background: 'rgba(255,255,255,0.05)',
                color: 'rgba(255,255,255,0.5)',
                border: '1px solid rgba(255,255,255,0.08)',
              }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              History
              {analyses.length > 0 && (
                <span
                  className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full text-[9px] font-black flex items-center justify-center text-black"
                  style={{ background: '#f5c518' }}
                >
                  {analyses.length}
                </span>
              )}
            </motion.button>

            {/* Logo */}
            <div className="flex items-center gap-2">
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center font-black text-xs text-black"
                style={{ background: '#f5c518' }}
              >
                TA
              </div>
              <span className="font-bold text-white text-sm tracking-tight hidden sm:block">
                Tennis<span style={{ color: '#f5c518' }}>AI</span>
              </span>
            </div>
          </div>

          {data && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={reset}
              className="text-xs font-semibold px-4 py-2 rounded-xl"
              style={{
                background: 'rgba(255,255,255,0.05)',
                color: 'rgba(255,255,255,0.5)',
                border: '1px solid rgba(255,255,255,0.08)',
              }}
            >
              New Analysis
            </motion.button>
          )}
        </div>
      </nav>

      {/* Sidebar */}
      <HistorySidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        analyses={analyses}
        onLoad={loadFromCache}
        onRemove={removeAnalysis}
        activeId={activeId}
      />

      <AnimatePresence mode="wait">
        {/* Upload page */}
        {!data && !loading && (
          <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -12 }}>
            <UploadZone onFile={handleFile} onUpload={handleUpload} file={file} />
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-md mx-auto px-4 pb-8"
              >
                <div
                  className="rounded-2xl p-4 text-sm flex items-start gap-3"
                  style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}
                >
                  <span className="text-red-400 mt-0.5 shrink-0">⚠</span>
                  <div>
                    <p className="font-semibold text-red-400 mb-1">Error</p>
                    <p style={{ color: 'rgba(239,68,68,0.75)' }}>{error}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {/* Processing */}
        {loading && (
          <motion.div key="processing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <ProcessingView progress={progress} statusMessage={statusMessage} />
          </motion.div>
        )}

        {/* Results */}
        {data && (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="max-w-7xl mx-auto px-4 py-8 space-y-6"
          >
            {/* "No video" banner + attach button */}
            <AnimatePresence>
              {needsVideoFile && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div
                    className="rounded-2xl p-4 flex items-center justify-between gap-4"
                    style={{ background: 'rgba(245,197,24,0.07)', border: '1px solid rgba(245,197,24,0.2)' }}
                  >
                    <div className="flex items-center gap-3">
                      <span style={{ color: '#f5c518' }}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </span>
                      <p className="text-sm" style={{ color: 'rgba(245,197,24,0.8)' }}>
                        Loaded from cache — stats & heatmaps are ready. Select the video file to enable live overlay.
                      </p>
                    </div>
                    <label
                      className="shrink-0 px-4 py-2 rounded-xl text-xs font-bold cursor-pointer text-black"
                      style={{ background: '#f5c518' }}
                    >
                      Select Video
                      <input
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => { if (e.target.files?.[0]) attachVideo(e.target.files[0]); }}
                      />
                    </label>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Summary pills */}
            <div className="flex gap-3 flex-wrap">
              <StatPill label="Total Bounces" value={bounces.length} delay={0} />
              <StatPill
                label="In / Out"
                value={`${inCount} / ${outCount}`}
                sub={inCount + outCount > 0 ? `${Math.round((inCount / (inCount + outCount)) * 100)}% accuracy` : undefined}
                delay={0.05}
              />
              <StatPill label="Avg Speed" value={avgSpeed > 0 ? `${avgSpeed.toFixed(1)} km/h` : '—'} delay={0.1} />
              <StatPill label="Peak Speed" value={maxSpeed > 0 ? `${maxSpeed.toFixed(1)} km/h` : '—'} delay={0.15} />
              <StatPill
                label="Frames"
                value={data.metadata.total_frames}
                sub={`${data.metadata.fps} fps`}
                delay={0.2}
              />
            </div>

            {/* Video + Court */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              <motion.div
                className="lg:col-span-2"
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.12 }}
              >
                {videoUrl ? (
                  <VideoOverlay videoUrl={videoUrl} data={data} onFrameUpdate={setCurrentFrame} />
                ) : (
                  <div
                    className="w-full rounded-2xl flex flex-col items-center justify-center gap-4 cursor-pointer"
                    style={{
                      aspectRatio: `${data.metadata.width} / ${data.metadata.height}`,
                      background: 'rgba(255,255,255,0.02)',
                      border: '2px dashed rgba(255,255,255,0.08)',
                    }}
                  >
                    <div
                      className="w-12 h-12 rounded-2xl flex items-center justify-center"
                      style={{ background: 'rgba(245,197,24,0.1)' }}
                    >
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#f5c518" strokeWidth="1.5">
                        <path strokeLinecap="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-sm font-medium" style={{ color: 'rgba(255,255,255,0.3)' }}>
                      Select video file to enable playback
                    </p>
                    <label
                      className="px-4 py-2 rounded-xl text-xs font-bold cursor-pointer text-black"
                      style={{ background: '#f5c518' }}
                    >
                      Select Video
                      <input
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => { if (e.target.files?.[0]) attachVideo(e.target.files[0]); }}
                      />
                    </label>
                  </div>
                )}
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.16 }}
              >
                <Court2D data={data} currentFrame={currentFrame} />
              </motion.div>
            </div>

            {/* Player comparison */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.22 }}
            >
              <div className="flex items-center gap-3 mb-4">
                <h2 className="text-sm font-bold text-white tracking-wide">Player Comparison</h2>
                <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.06)' }} />
              </div>
              <div className="flex gap-4 items-stretch">
                <PlayerCard
                  side="top"
                  stats={topStats}
                  bounces={bounces}
                  ballSpeeds={ballSpeeds}
                />

                {/* VS */}
                <div className="flex flex-col items-center justify-center gap-2 shrink-0 py-4">
                  <div className="w-px flex-1" style={{ background: 'rgba(255,255,255,0.05)' }} />
                  <div
                    className="text-[10px] font-black rounded-full w-8 h-8 flex items-center justify-center"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      color: 'rgba(255,255,255,0.2)',
                      border: '1px solid rgba(255,255,255,0.07)',
                    }}
                  >
                    VS
                  </div>
                  <div className="w-px flex-1" style={{ background: 'rgba(255,255,255,0.05)' }} />
                </div>

                <PlayerCard
                  side="bottom"
                  stats={bottomStats}
                  bounces={bounces}
                  ballSpeeds={ballSpeeds}
                />
              </div>
            </motion.div>

            {/* Speed chart */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.28 }}
            >
              <SpeedChart speedData={speedData} />
            </motion.div>

            {/* Color legend */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.32 }}
              className="flex items-center gap-6 text-xs pb-8"
              style={{ color: 'rgba(255,255,255,0.25)' }}
            >
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 rounded-full" style={{ background: P1 }} />
                Player 1 (Top)
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 rounded-full" style={{ background: P2 }} />
                Player 2 (Bottom)
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 rounded-full" style={{ background: '#f5c518' }} />
                Ball trajectory
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
