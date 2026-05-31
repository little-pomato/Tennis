import React, { useEffect, useRef, useMemo } from 'react';

// Real court dimensions (metres)
const COURT = { w: 10.97, h: 23.77, singles_w: 8.23, service_line: 6.4 };

// Canvas dimensions
const CW = 140;
const CH = Math.round(CW * (COURT.h / COURT.w));
const PAD_X = 10;
const PAD_Y = 12;
const IW = CW - PAD_X * 2; // inner court width
const IH = CH - PAD_Y * 2;

const px = (x: number) => PAD_X + (x / COURT.w) * IW;
const py = (y: number) => PAD_Y + (y / COURT.h) * IH;
const MARGIN = (COURT.w - COURT.singles_w) / 2;

// Convert hex "#rrggbb" to "r,g,b"
function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

interface Props {
  bounces: Array<{ pos_2d: [number, number]; status: string; frame: number }>;
  ballSpeeds: Array<{ start: number; end: number; side: string; speed_kmh: number }>;
  playerSide: 'top' | 'bottom';
  playerColor: string; // hex
  empty?: boolean;
}

const ShotHeatmap: React.FC<Props> = ({ bounces, ballSpeeds, playerSide, playerColor, empty }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Sorted speeds for consistent lookup (same approach as VideoOverlay/Court2D)
  const sortedSpeeds = useMemo(
    () => [...ballSpeeds].sort((a, b) => a.end - b.end),
    [ballSpeeds]
  );

  // Filter to bounces caused by this player.
  // Uses end-frame matching: the event whose end === bounce.frame owns that bounce.
  const playerBounces = useMemo(() => {
    if (empty) return [];
    return bounces.filter((b) => {
      if (!b.pos_2d) return false;
      if (sortedSpeeds.length > 0) {
        // Find the event whose bounce (end) matches this bounce frame
        const exact = sortedSpeeds.find((s) => s.end === b.frame);
        if (exact) return exact.side === playerSide;
        // Fallback: nearest event by end frame
        const nearest = [...sortedSpeeds].sort(
          (a, c) => Math.abs(a.end - b.frame) - Math.abs(c.end - b.frame)
        )[0];
        if (nearest && Math.abs(nearest.end - b.frame) <= 3) return nearest.side === playerSide;
      }
      // No analytics: physics-based fallback
      // P1 (top) shots land on bottom half; P2 (bottom) shots land on top half
      const isTopHalf = b.pos_2d[1] < COURT.h / 2;
      return playerSide === 'top' ? !isTopHalf : isTopHalf;
    });
  }, [bounces, sortedSpeeds, playerSide, empty]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // ---- 1. Dark court background ----
    ctx.clearRect(0, 0, CW, CH);
    ctx.fillStyle = '#0e2910';
    ctx.fillRect(0, 0, CW, CH);

    // Court surface
    ctx.fillStyle = '#183d14';
    ctx.fillRect(PAD_X, PAD_Y, IW, IH);

    // ---- 2. Heatmap layer using radial gradient blobs ----
    if (playerBounces.length > 0) {
      const BLOB_R = Math.max(10, Math.round(IW * 0.14));
      const rgb = hexToRgb(playerColor);

      // Use 'screen' composite: blobs accumulate into bright hot spots
      ctx.globalCompositeOperation = 'screen';

      playerBounces.forEach((b) => {
        const bx = px(b.pos_2d[0]);
        const by = py(b.pos_2d[1]);
        const grad = ctx.createRadialGradient(bx, by, 0, bx, by, BLOB_R);
        grad.addColorStop(0, `rgba(${rgb}, 0.55)`);
        grad.addColorStop(0.4, `rgba(${rgb}, 0.25)`);
        grad.addColorStop(1, `rgba(${rgb}, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(bx, by, BLOB_R, 0, Math.PI * 2);
        ctx.fill();
      });

      // Second pass: brighter core for high-density spots
      ctx.globalCompositeOperation = 'screen';
      playerBounces.forEach((b) => {
        const bx = px(b.pos_2d[0]);
        const by = py(b.pos_2d[1]);
        const grad = ctx.createRadialGradient(bx, by, 0, bx, by, BLOB_R * 0.45);
        grad.addColorStop(0, `rgba(255, 255, 255, 0.18)`);
        grad.addColorStop(1, `rgba(255, 255, 255, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(bx, by, BLOB_R * 0.45, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.globalCompositeOperation = 'source-over';
    }

    // ---- 3. Court lines on top ----
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;

    // Outer boundary
    ctx.strokeRect(px(0), py(0), IW, IH);

    // Singles lines
    ctx.beginPath();
    ctx.moveTo(px(MARGIN), py(0)); ctx.lineTo(px(MARGIN), py(COURT.h));
    ctx.moveTo(px(COURT.w - MARGIN), py(0)); ctx.lineTo(px(COURT.w - MARGIN), py(COURT.h));
    ctx.stroke();

    // Service lines
    ctx.beginPath();
    ctx.moveTo(px(MARGIN), py(COURT.h / 2 - COURT.service_line));
    ctx.lineTo(px(COURT.w - MARGIN), py(COURT.h / 2 - COURT.service_line));
    ctx.moveTo(px(MARGIN), py(COURT.h / 2 + COURT.service_line));
    ctx.lineTo(px(COURT.w - MARGIN), py(COURT.h / 2 + COURT.service_line));
    ctx.stroke();

    // Center service line
    ctx.beginPath();
    ctx.moveTo(px(COURT.w / 2), py(COURT.h / 2 - COURT.service_line));
    ctx.lineTo(px(COURT.w / 2), py(COURT.h / 2 + COURT.service_line));
    ctx.stroke();

    // Net
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(px(-0.1) - 1, py(COURT.h / 2));
    ctx.lineTo(px(COURT.w + 0.1) + 1, py(COURT.h / 2));
    ctx.stroke();
    ctx.setLineDash([]);

    // ---- 4. Empty state text ----
    if (playerBounces.length === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No shot data', CW / 2, CH / 2);
      ctx.textAlign = 'left';
    }
  }, [playerBounces, playerColor]);

  return (
    <canvas
      ref={canvasRef}
      width={CW}
      height={CH}
      className="rounded-lg"
      style={{ imageRendering: 'crisp-edges' }}
    />
  );
};

export default ShotHeatmap;
