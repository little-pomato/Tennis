import React, { useRef, useEffect, useMemo } from 'react';

interface Props {
  videoUrl: string;
  data: any;
  onFrameUpdate?: (frame: number) => void;
}

// High-visibility trail colors — readable against any court surface
const COLOR_P1      = 'rgb(255, 255, 255)';   // white
const COLOR_P2      = 'rgb(10, 186, 181)';    // Tiffany green
const COLOR_UNKNOWN = 'rgb(160, 240, 230)';   // pale Tiffany (pre-first hit)

const VideoOverlay: React.FC<Props> = ({ videoUrl, data, onFrameUpdate }) => {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(null);

  const fps            = data.metadata.fps;
  const originalWidth  = data.metadata.width;
  const originalHeight = data.metadata.height;
  const totalFrames    = data.metadata.total_frames;

  const ballSpeeds: any[] = useMemo(
    () => [...(data.results.analytics?.ball_speeds ?? [])].sort((a, b) => a.start - b.start),
    [data]
  );
  const ballOwnership: any[] = data.results.analytics?.ball_ownership ?? [];

  const getBallColor = (frameIdx: number): string => {
    const ownership = ballOwnership[frameIdx];
    if (ownership?.owner)
      return ownership.owner === 'bottom' ? COLOR_P2 : COLOR_P1;

    let activeEvent = null;
    for (const evt of ballSpeeds) {
      if (evt.start <= frameIdx) activeEvent = evt;
      else break;
    }
    if (activeEvent)
      return activeEvent.side === 'bottom' ? COLOR_P2 : COLOR_P1;

    return COLOR_UNKNOWN;
  };

  const getInterpolatedBall = (preciseFrame: number) => {
    const f1 = Math.floor(preciseFrame);
    const f2 = f1 + 1;
    const t  = preciseFrame - f1;
    const p1 = data.results.ball_track[f1];
    const p2 = data.results.ball_track[f2];
    if (p1 && p1.x !== null && p2 && p2.x !== null)
      return { x: p1.x + (p2.x - p1.x) * t, y: p1.y + (p2.y - p1.y) * t };
    return p1 || null;
  };

  const getBounceVideoPoint = (bounce: any) => {
    if (bounce.pos_img && bounce.pos_img[0] !== null)
      return { x: bounce.pos_img[0], y: bounce.pos_img[1] };
    const tracked = data.results.ball_track[bounce.frame];
    if (tracked && tracked.x !== null)
      return { x: tracked.x, y: tracked.y };
    return null;
  };

  // Parse "rgb(r,g,b)" → [r,g,b]
  const parseRgb = (c: string): [number, number, number] => {
    const m = c.match(/\d+/g);
    return m ? [+m[0], +m[1], +m[2]] : [200, 200, 200];
  };

  const renderFrame = () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const preciseFrame = video.currentTime * fps;
    const frameIdx     = Math.floor(preciseFrame);
    if (onFrameUpdate) onFrameUpdate(frameIdx);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (frameIdx >= totalFrames) {
      requestRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const scaleX = canvas.width  / originalWidth;
    const scaleY = canvas.height / originalHeight;

    // 1. Ball trail — warm gradient fading backward
    const maxHistory = 28;
    const startJ     = Math.max(1, frameIdx - maxHistory);
    const trailColor = getBallColor(frameIdx);
    const [tr, tg, tb] = parseRgb(trailColor);

    ctx.lineCap  = 'round';
    ctx.lineJoin = 'round';

    for (let j = startJ; j <= frameIdx; j++) {
      const p1 = data.results.ball_track[j - 1];
      const p2 = data.results.ball_track[j];
      if (p1 && p2 && p1.x !== null && p2.x !== null) {
        const age    = frameIdx - j;
        const alpha  = Math.max(0, 1.0 - age / maxHistory);
        const width  = 1.0 + 2.5 * (1 - age / maxHistory);
        ctx.lineWidth   = width;
        ctx.strokeStyle = `rgba(${tr},${tg},${tb},${alpha * 0.72})`;
        ctx.beginPath();
        ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
        ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
        ctx.stroke();
      }
    }

    // 2. Ball — warm yellow-green with amber glow
    const ball = getInterpolatedBall(preciseFrame);
    if (ball && ball.x !== null) {
      const bx = ball.x * scaleX;
      const by = ball.y * scaleY;

      // Outer Tiffany glow ring
      ctx.beginPath();
      ctx.arc(bx, by, 9, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(10,186,181,0.32)';
      ctx.lineWidth   = 3;
      ctx.stroke();

      // Mid ring — white
      ctx.beginPath();
      ctx.arc(bx, by, 6, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth   = 1.5;
      ctx.stroke();

      // Ball fill — white core with Tiffany glow
      ctx.beginPath();
      ctx.arc(bx, by, 4.5, 0, Math.PI * 2);
      ctx.shadowBlur  = 14;
      ctx.shadowColor = '#0ABAB5';
      ctx.fillStyle   = '#FFFFFF';
      ctx.fill();
      ctx.shadowBlur  = 0;

      // Specular highlight
      ctx.beginPath();
      ctx.arc(bx - 1.2, by - 1.2, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fill();
    }

    // 3. Players — warm labeled bounding boxes
    const players = data.results.players[frameIdx];
    if (players) {
      const drawPlayer = (bbox: number[], color: string, label: string, labelBottom: boolean) => {
        const [r, g, b] = parseRgb(color);
        const x = bbox[0] * scaleX, y = bbox[1] * scaleY;
        const w = (bbox[2] - bbox[0]) * scaleX, h = (bbox[3] - bbox[1]) * scaleY;

        // Rounded rect box with glow
        ctx.strokeStyle = `rgba(${r},${g},${b},0.92)`;
        ctx.lineWidth   = 2.5;
        ctx.shadowBlur  = 8;
        ctx.shadowColor = `rgba(${r},${g},${b},0.5)`;
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur  = 0;

        // Label badge
        const lx = x;
        const ly = labelBottom ? (bbox[3] * scaleY) + 3 : y - 20;
        ctx.fillStyle = `rgba(${r},${g},${b},0.85)`;
        ctx.fillRect(lx - 1, ly, ctx.measureText(label).width + 14, 17);
        ctx.font      = 'bold 10px "Space Mono", monospace';
        ctx.fillStyle = 'white';
        ctx.fillText(label, lx + 6, ly + 12);
      };
      players.top.forEach((bbox: number[])    => drawPlayer(bbox, COLOR_P1, 'P1 TOP', false));
      players.bottom.forEach((bbox: number[]) => drawPlayer(bbox, COLOR_P2, 'P2 BOTTOM', true));
    }

    // 4. Bounce markers — warm amber expanding rings
    data.results.bounces.forEach((bounce: any) => {
      const diff = frameIdx - bounce.frame;
      if (diff < 0 || diff >= 42) return;

      const videoPoint = getBounceVideoPoint(bounce);
      if (!videoPoint) return;

      const bx     = videoPoint.x * scaleX;
      const by     = videoPoint.y * scaleY;
      const alpha  = 1 - diff / 42;
      const isOut  = bounce.status.includes('Out');
      const [r, g, b] = parseRgb(getBallColor(bounce.frame));

      const markerR = diff < 10 ? 7 + 3 * (1 - diff / 10) : 5;

      // Expanding amber shockwave rings (three layers)
      const ringProgress = diff / 42;

      // Ring 1 — fast inner, white
      ctx.beginPath();
      ctx.arc(bx, by, markerR + ringProgress * 28, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${(1 - ringProgress) * 0.8})`;
      ctx.lineWidth   = 2.5;
      ctx.stroke();

      // Ring 2 — medium, Tiffany, delayed
      if (diff > 3) {
        const r2p = (diff - 3) / 42;
        ctx.beginPath();
        ctx.arc(bx, by, markerR + r2p * 46, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(10,186,181,${(1 - r2p) * 0.55})`;
        ctx.lineWidth   = 1.5;
        ctx.stroke();
      }

      // Dark shadow halo
      ctx.beginPath();
      ctx.arc(bx, by, markerR + 5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,0,0,${0.28 * alpha})`;
      ctx.fill();

      // Dot marker
      ctx.beginPath();
      ctx.arc(bx, by, markerR, 0, Math.PI * 2);
      if (isOut) {
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha * 0.95})`;
        ctx.lineWidth   = 2.5;
        ctx.stroke();
        const xr = markerR * 0.55;
        ctx.beginPath();
        ctx.moveTo(bx - xr, by - xr); ctx.lineTo(bx + xr, by + xr);
        ctx.moveTo(bx + xr, by - xr); ctx.lineTo(bx - xr, by + xr);
        ctx.stroke();
      } else {
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha * 0.9})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(255,248,220,${alpha * 0.6})`;
        ctx.lineWidth   = 1.5;
        ctx.stroke();
      }

      // Label (fades quickly)
      if (diff < 22) {
        const labelAlpha = 1 - diff / 22;
        const label      = isOut ? bounce.status.replace('Out - ', '') : 'In';
        ctx.font = 'bold 11px "Space Mono", monospace';
        const mw = ctx.measureText(label).width;
        const lx = bx + 14, ly = by - 20;
        ctx.fillStyle = `rgba(30,41,56,${0.65 * labelAlpha})`;
        ctx.fillRect(lx - 5, ly - 14, mw + 12, 20);
        ctx.fillStyle = `rgba(255,248,220,${labelAlpha})`;
        ctx.fillText(label, lx + 1, ly);
      }
    });

    requestRef.current = requestAnimationFrame(renderFrame);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(renderFrame);
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, [data]);

  return (
    <div
      className="relative w-full max-w-5xl mx-auto rounded-3xl overflow-hidden"
      style={{ boxShadow: 'var(--shadow-raised-lg)' }}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        className="w-full block"
        controls
        muted
        playsInline
        style={{ borderRadius: 'inherit' }}
      />
      <canvas
        ref={canvasRef}
        width={originalWidth}
        height={originalHeight}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
    </div>
  );
};

export default VideoOverlay;
