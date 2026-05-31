import React, { useRef, useEffect, useMemo } from 'react';

interface Props {
  videoUrl: string;
  data: any;
  onFrameUpdate?: (frame: number) => void;
}

const VideoOverlay: React.FC<Props> = ({ videoUrl, data, onFrameUpdate }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(null);

  const fps = data.metadata.fps;
  const originalWidth = data.metadata.width;
  const originalHeight = data.metadata.height;
  const totalFrames = data.metadata.total_frames;

  // Sort ball_speeds by bounce frame once; used for trajectory coloring.
  // Key insight: we use the NEXT bounce's event to color each frame segment,
  // not the [start, end] range — because `start` (hit frame) is estimated and
  // can be off, but `end` (bounce frame) is always exact. This ensures the
  // entire trajectory between two bounces shows one solid, correct color.
  const ballSpeeds: any[] = useMemo(
    () => [...(data.results.analytics?.ball_speeds ?? [])].sort((a, b) => a.end - b.end),
    [data]
  );

  const getBallColor = (frameIdx: number): string => {
    // Find the first event whose bounce (end) is >= frameIdx.
    // Everything before that bounce belongs to that hitter.
    for (const evt of ballSpeeds) {
      if (evt.end >= frameIdx) {
        return evt.side === 'bottom' ? 'rgb(59, 130, 246)' : 'rgb(239, 68, 68)';
      }
    }
    // After the last known bounce: keep last hitter's color
    if (ballSpeeds.length > 0) {
      const last = ballSpeeds[ballSpeeds.length - 1];
      return last.side === 'bottom' ? 'rgb(59, 130, 246)' : 'rgb(239, 68, 68)';
    }
    return 'rgb(245, 197, 24)'; // no analytics → yellow
  };

  // Interpolate ball position for sub-frame accuracy
  const getInterpolatedBall = (preciseFrame: number) => {
    const f1 = Math.floor(preciseFrame);
    const f2 = f1 + 1;
    const t = preciseFrame - f1;

    const p1 = data.results.ball_track[f1];
    const p2 = data.results.ball_track[f2];

    if (p1 && p1.x !== null && p2 && p2.x !== null) {
      return {
        x: p1.x + (p2.x - p1.x) * t,
        y: p1.y + (p2.y - p1.y) * t
      };
    }
    return p1 || null;
  };

  const renderFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // PRECISE CALCULATION
    const preciseFrame = video.currentTime * fps;
    const frameIdx = Math.floor(preciseFrame);
    
    if (onFrameUpdate) onFrameUpdate(frameIdx);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (frameIdx >= totalFrames) {
      requestRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const scaleX = canvas.width / originalWidth;
    const scaleY = canvas.height / originalHeight;

    // 1. Draw Trajectory (Enhanced 'Comet' Trail)
    const maxHistory = 30;
    const startJ = Math.max(1, frameIdx - maxHistory);
    
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let j = startJ; j <= frameIdx; j++) {
      const p1 = data.results.ball_track[j - 1];
      const p2 = data.results.ball_track[j];
      
      if (p1 && p2 && p1.x !== null && p2.x !== null) {
        const age = frameIdx - j;
        const alpha = Math.max(0, 1.0 - (age / maxHistory));
        const color = getBallColor(j);
        
        // Tapered width: thicker near the ball, thinner at the tail
        ctx.lineWidth = 1.0 + (3.0 * (1 - age / maxHistory));
        
        const rgba = color.replace('rgb', 'rgba').replace(')', `, ${alpha * 0.7})`);
        
        ctx.beginPath();
        ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
        ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
        ctx.strokeStyle = rgba;
        ctx.stroke();
      }
    }

    // 2. Draw Current Ball (High-precision with subtle glow)
    const ball = getInterpolatedBall(preciseFrame);
    if (ball && ball.x !== null) {
      const bx = ball.x * scaleX;
      const by = ball.y * scaleY;

      // Inner glow
      ctx.beginPath();
      ctx.arc(bx, by, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#0ff';
      ctx.fill();
      ctx.shadowBlur = 0;
      
      // Outer pulse
      ctx.beginPath();
      ctx.arc(bx, by, 6.5, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // 3. Draw Players
    const players = data.results.players[frameIdx];
    if (players) {
      players.top.forEach((bbox: number[]) => {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox[0] * scaleX, bbox[1] * scaleY, (bbox[2] - bbox[0]) * scaleX, (bbox[3] - bbox[1]) * scaleY);
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 11px Inter';
        ctx.fillText("TOP PLAYER", bbox[0] * scaleX, bbox[1] * scaleY - 5);
      });

      players.bottom.forEach((bbox: number[]) => {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox[0] * scaleX, bbox[1] * scaleY, (bbox[2] - bbox[0]) * scaleX, (bbox[3] - bbox[1]) * scaleY);
        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 11px Inter';
        ctx.fillText("BOTTOM PLAYER", bbox[0] * scaleX, (bbox[3] * scaleY) + 15);
      });
    }

    // 4. Draw Bounces (Shockwave effect)
    data.results.bounces.forEach((bounce: any) => {
      const diff = frameIdx - bounce.frame;
      if (diff >= 0 && diff < 45) {
        const isFresh = diff < 8;
        const alpha = 1 - (diff / 45);
        const isOut = bounce.status.includes('Out');
        const baseColor = getBallColor(bounce.frame);
        const colorStr = baseColor.replace('rgb(', '').replace(')', '');
        const bx = bounce.pos_2d[0] * scaleX;
        const by = bounce.pos_2d[1] * scaleY;
        
        // Impact Shockwave
        if (isFresh) {
          const rippleRadius = (diff / 8) * 45;
          const rippleAlpha = 1 - (diff / 8);
          
          ctx.beginPath();
          ctx.arc(bx, by, rippleRadius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(${colorStr}, ${rippleAlpha})`;
          ctx.lineWidth = 3;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(bx, by, rippleRadius * 0.7, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(255, 255, 255, ${rippleAlpha * 0.6})`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Marker
        ctx.beginPath();
        const markerRadius = isFresh ? 12 + (8 * (1 - diff / 8)) : 6;
        ctx.arc(bx, by, markerRadius, 0, Math.PI * 2);
        
        if (isOut) {
          ctx.strokeStyle = `rgba(${colorStr}, ${alpha})`;
          ctx.lineWidth = isFresh ? 4 : 2;
          ctx.stroke();
          
          // Cross
          const r = markerRadius * 0.7;
          ctx.beginPath();
          ctx.moveTo(bx - r, by - r); ctx.lineTo(bx + r, by + r);
          ctx.moveTo(bx + r, by - r); ctx.lineTo(bx - r, by + r);
          ctx.stroke();
        } else {
          ctx.fillStyle = `rgba(${colorStr}, ${alpha})`;
          ctx.fill();
          ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.5})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        
        if (diff < 15) {
          ctx.fillStyle = `rgba(255, 255, 255, ${1 - diff / 15})`;
          ctx.font = 'bold 16px Inter';
          ctx.fillText(bounce.status, bx + 25, by + 5);
        }
      }
    });

    requestRef.current = requestAnimationFrame(renderFrame);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(renderFrame);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [data]);

  return (
    <div className="relative w-full max-w-5xl mx-auto bg-black rounded-2xl overflow-hidden shadow-2xl border border-gray-800">
      <video ref={videoRef} src={videoUrl} className="w-full block" controls muted playsInline />
      <canvas ref={canvasRef} width={originalWidth} height={originalHeight} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
    </div>
  );
};

export default VideoOverlay;
