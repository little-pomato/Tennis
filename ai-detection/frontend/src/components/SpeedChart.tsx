import React from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

interface SpeedPoint {
  frame: number;
  speed: number;
}

interface Props {
  speedData: SpeedPoint[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div
        className="px-3 py-2 rounded-xl text-sm"
        style={{ background: '#0f1629', border: '1px solid rgba(255,255,255,0.08)' }}
      >
        <p style={{ color: 'rgba(255,255,255,0.4)' }} className="text-xs mb-0.5">Frame {label}</p>
        <p className="font-bold" style={{ color: '#f5c518' }}>{payload[0].value.toFixed(1)} km/h</p>
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
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.1 }}
      className="rounded-2xl p-6"
      style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.07)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div className="flex items-start justify-between mb-6">
        <div>
          <h3 className="font-bold text-white text-base">Ball Speed Timeline</h3>
          <p className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.35)' }}>
            Velocity across all tracked frames
          </p>
        </div>
        <div className="flex gap-4">
          <div className="text-right">
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>Average</p>
            <p className="text-sm font-bold" style={{ color: '#f5c518' }}>{avg.toFixed(1)} km/h</p>
          </div>
          <div className="text-right">
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>Peak</p>
            <p className="text-sm font-bold text-white">{max.toFixed(1)} km/h</p>
          </div>
        </div>
      </div>

      {speedData.length > 0 ? (
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={speedData} margin={{ top: 4, right: 4, bottom: 0, left: -10 }}>
              <defs>
                <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f5c518" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f5c518" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="frame"
                stroke="transparent"
                tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
                axisLine={false}
              />
              <YAxis
                stroke="transparent"
                tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
                axisLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(245,197,24,0.3)', strokeWidth: 1 }} />
              <Area
                type="monotone"
                dataKey="speed"
                stroke="#f5c518"
                strokeWidth={2}
                fill="url(#speedGradient)"
                dot={false}
                activeDot={{ r: 4, fill: '#f5c518', strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div
          className="h-52 flex items-center justify-center rounded-xl text-sm"
          style={{ background: 'rgba(255,255,255,0.02)', color: 'rgba(255,255,255,0.25)' }}
        >
          No speed data available
        </div>
      )}
    </motion.div>
  );
};

export default SpeedChart;
