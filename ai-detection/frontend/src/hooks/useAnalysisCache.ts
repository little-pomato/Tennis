import { useState } from 'react';

export interface CachedAnalysis {
  id: string;
  filename: string;
  analyzedAt: string;
  data: any;
  stats: {
    totalBounces: number;
    inCount: number;
    avgSpeed: number;
    fps: number;
    totalFrames: number;
  };
}

const STORAGE_KEY = 'tennisai_analyses';
const MAX_ENTRIES = 10;

function loadCache(): CachedAnalysis[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function persistCache(entries: CachedAnalysis[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Quota exceeded — drop oldest entry and retry once
    const trimmed = entries.slice(0, -1);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed)); } catch { /* ignore */ }
  }
}

export function useAnalysisCache() {
  const [analyses, setAnalyses] = useState<CachedAnalysis[]>(loadCache);

  const saveAnalysis = (filename: string, data: any): string => {
    const bounces: any[] = data.results?.bounces ?? [];
    const speeds: any[] = data.results?.analytics?.ball_speeds ?? [];
    const avg = speeds.length > 0
      ? speeds.reduce((a: number, s: any) => a + (s.speed_kmh ?? 0), 0) / speeds.length
      : 0;

    const entry: CachedAnalysis = {
      id: crypto.randomUUID(),
      filename,
      analyzedAt: new Date().toISOString(),
      data,
      stats: {
        totalBounces: bounces.length,
        inCount: bounces.filter((b) => b.status === 'In').length,
        avgSpeed: avg,
        fps: data.metadata?.fps ?? 0,
        totalFrames: data.metadata?.total_frames ?? 0,
      },
    };

    setAnalyses((prev) => {
      const next = [entry, ...prev].slice(0, MAX_ENTRIES);
      persistCache(next);
      return next;
    });
    return entry.id;
  };

  const removeAnalysis = (id: string) => {
    setAnalyses((prev) => {
      const next = prev.filter((a) => a.id !== id);
      persistCache(next);
      return next;
    });
  };

  return { analyses, saveAnalysis, removeAnalysis };
}
