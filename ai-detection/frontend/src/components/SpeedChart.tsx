import React from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

interface SpeedPoint { frame: number; speed: number; }
interface Props { speedData: SpeedPoint[]; }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div
        className="px-3 py-2 rounded-xl text-sm"
        style={{
          background: 'var(--surface)',
          boxShadow: 'var(--shadow-raised-sm)',
          border: '1.5px solid rgba(0,102,102,0.15)',
        }}
      >
        <p
          className="text-xs mb-0.5"
          style={{ color: 'var(--text-muted)', fontFamily: "'Space Mono', monospace" }}
        >
          Frame {label}
        </p>
        <p
          className="font-bold"
          style={{ color: 'var(--primary)', fontFamily: "'JetBrains Mono', monospace" }}
        >
          {payload[0].value.toFixed(1)} km/h
        </p>
      </div>
    );
  }
  return null;
};

const SpeedChart: React.FC<Props> = ({ speedData }) => {
  const avg = speedData.length > 0
    ? speedData.reduce((a, b) => a + b.speed, 0) / speedData.length
    : 0;
  const max = speedData.length > 0
    ? Math.max(...speedData.map((d) => d.speed))
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.1 }}
      className="rounded-3xl p-6"
      style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-raised)' }}
    >
      <div className="flex items-start justify-between mb-6">
        <div>
          <h3
            className="font-bold text-base"
            style={{ color: 'var(--text)', fontFamily: "'Space Mono', monospace" }}
          >
            Ball Speed Timeline
          </h3>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            Velocity across all tracked frames
          </p>
        </div>

        <div className="flex gap-5">
          <div
            className="text-right rounded-2xl px-4 py-2.5"
            style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
          >
            <p className="text-[10px] font-bold" style={{ color: 'var(--text-subtle)', fontFamily: "'Space Mono', monospace" }}>
              Average
            </p>
            <p
              className="text-sm font-bold font-mono-nums"
              style={{ color: 'var(--primary)', fontFamily: "'JetBrains Mono', monospace" }}
            >
              {avg.toFixed(1)} km/h
            </p>
          </div>
          <div
            className="text-right rounded-2xl px-4 py-2.5"
            style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
          >
            <p className="text-[10px] font-bold" style={{ color: 'var(--text-subtle)', fontFamily: "'Space Mono', monospace" }}>
              Peak
            </p>
            <p
              className="text-sm font-bold font-mono-nums"
              style={{ color: 'var(--text)', fontFamily: "'JetBrains Mono', monospace" }}
            >
              {max.toFixed(1)} km/h
            </p>
          </div>
        </div>
      </div>

      {speedData.length > 0 ? (
        <div
          className="h-52 rounded-2xl p-3"
          style={{ background: 'var(--surface)', boxShadow: 'var(--shadow-pressed-sm)' }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={speedData} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
              <defs>
                <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#006666" stopOpacity={0.25} />
                  <stop offset="50%" stopColor="#FE9900" stopOpacity={0.12} />
                  <stop offset="95%" stopColor="#006666" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,56,0.06)" />
              <XAxis
                dataKey="frame"
                stroke="transparent"
                tick={{ fill: 'var(--text-subtle)', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
              />
              <YAxis
                stroke="transparent"
                tick={{ fill: 'var(--text-subtle)', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
              />
              <Tooltip
                content={<CustomTooltip />}
                cursor={{ stroke: 'rgba(0,102,102,0.3)', strokeWidth: 1.5 }}
              />
              <Area
                type="monotone"
                dataKey="speed"
                stroke="var(--primary)"
                strokeWidth={2.5}
                fill="url(#speedGradient)"
                dot={false}
                activeDot={{ r: 5, fill: 'var(--primary)', strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div
          className="h-52 flex items-center justify-center rounded-2xl text-sm"
          style={{
            background: 'var(--surface)',
            boxShadow: 'var(--shadow-pressed-sm)',
            color: 'var(--text-subtle)',
            fontFamily: "'Space Mono', monospace",
          }}
        >
          No speed data available
        </div>
      )}
    </motion.div>
  );
};

export default SpeedChart;
