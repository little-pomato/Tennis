import React, { useEffect, useRef, useMemo } from 'react';

const COURT = { w: 10.97, h: 23.77, singles_w: 8.23, service_line: 6.4 };
const CW    = 140;
const CH    = Math.round(CW * (COURT.h / COURT.w));
const PAD_X = 10;
const PAD_Y = 12;
const IW    = CW - PAD_X * 2;
const IH    = CH - PAD_Y * 2;

const px = (x: number) => PAD_X + (x / COURT.w) * IW;
const py = (y: number) => PAD_Y + (y / COURT.h) * IH;
const MARGIN = (COURT.w - COURT.singles_w) / 2;

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
  playerColor: string;
  empty?: boolean;
}

const ShotHeatmap: React.FC<Props> = ({ bounces, ballSpeeds, playerSide, playerColor, empty }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const sortedSpeeds = useMemo(
    () => [...ballSpeeds].sort((a, b) => a.end - b.end),
    [ballSpeeds]
  );

  const playerBounces = useMemo(() => {
    if (empty) return [];
    return bounces.filter((b) => {
      if (!b.pos_2d) return false;
      if (sortedSpeeds.length > 0) {
        const exact = sortedSpeeds.find((s) => s.end === b.frame);
        if (exact) return exact.side === playerSide;
        const nearest = [...sortedSpeeds].sort(
          (a, c) => Math.abs(a.end - b.frame) - Math.abs(c.end - b.frame)
        )[0];
        if (nearest && Math.abs(nearest.end - b.frame) <= 3) return nearest.side === playerSide;
      }
      const isTopHalf = b.pos_2d[1] < COURT.h / 2;
      return playerSide === 'top' ? !isTopHalf : isTopHalf;
    });
  }, [bounces, sortedSpeeds, playerSide, empty]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CW, CH);

    // Clay court background
    ctx.fillStyle = '#8C4B22';
    ctx.fillRect(0, 0, CW, CH);

    // Clay playing surface
    ctx.fillStyle = '#B86E38';
    ctx.fillRect(PAD_X, PAD_Y, IW, IH);

    // Heatmap blobs
    if (playerBounces.length > 0) {
      const BLOB_R = Math.max(10, Math.round(IW * 0.14));
      const rgb    = hexToRgb(playerColor);

      ctx.globalCompositeOperation = 'screen';

      playerBounces.forEach((b) => {
        const bx   = px(b.pos_2d[0]);
        const by   = py(b.pos_2d[1]);
        const grad = ctx.createRadialGradient(bx, by, 0, bx, by, BLOB_R);
        grad.addColorStop(0,   `rgba(${rgb}, 0.62)`);
        grad.addColorStop(0.4, `rgba(${rgb}, 0.28)`);
        grad.addColorStop(1,   `rgba(${rgb}, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(bx, by, BLOB_R, 0, Math.PI * 2);
        ctx.fill();
      });

      // Bright core for dense spots
      playerBounces.forEach((b) => {
        const bx   = px(b.pos_2d[0]);
        const by   = py(b.pos_2d[1]);
        const grad = ctx.createRadialGradient(bx, by, 0, bx, by, BLOB_R * 0.45);
        grad.addColorStop(0, `rgba(255,245,220,0.22)`);
        grad.addColorStop(1, `rgba(255,245,220,0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(bx, by, BLOB_R * 0.45, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.globalCompositeOperation = 'source-over';
    }

    // Court lines — warm white
    ctx.strokeStyle = 'rgba(255,248,220,0.88)';
    ctx.lineWidth   = 1;

    ctx.strokeRect(px(0), py(0), IW, IH);

    ctx.beginPath();
    ctx.moveTo(px(MARGIN),           py(0)); ctx.lineTo(px(MARGIN),           py(COURT.h));
    ctx.moveTo(px(COURT.w - MARGIN), py(0)); ctx.lineTo(px(COURT.w - MARGIN), py(COURT.h));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(px(MARGIN), py(COURT.h / 2 - COURT.service_line));
    ctx.lineTo(px(COURT.w - MARGIN), py(COURT.h / 2 - COURT.service_line));
    ctx.moveTo(px(MARGIN), py(COURT.h / 2 + COURT.service_line));
    ctx.lineTo(px(COURT.w - MARGIN), py(COURT.h / 2 + COURT.service_line));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(px(COURT.w / 2), py(COURT.h / 2 - COURT.service_line));
    ctx.lineTo(px(COURT.w / 2), py(COURT.h / 2 + COURT.service_line));
    ctx.stroke();

    // Net
    ctx.strokeStyle = 'rgba(255,248,220,0.55)';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(px(-0.1) - 1, py(COURT.h / 2));
    ctx.lineTo(px(COURT.w + 0.1) + 1, py(COURT.h / 2));
    ctx.stroke();
    ctx.setLineDash([]);

    // Empty state
    if (playerBounces.length === 0) {
      ctx.fillStyle = 'rgba(255,248,220,0.35)';
      ctx.font      = '8px "Space Mono", monospace';
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
      className="rounded-xl"
      style={{ imageRendering: 'crisp-edges' }}
    />
  );
};

export default ShotHeatmap;
