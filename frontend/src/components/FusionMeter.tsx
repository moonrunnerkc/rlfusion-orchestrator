// frontend/src/components/FusionMeter.tsx
// 2-path CAG + Graph meter. The policy decides dominance. The UI obeys.

import { motion } from 'framer-motion';
import type { Weights } from '../types/contracts';

interface FusionMeterProps {
  weights: Weights;
}

export default function FusionMeter({ weights }: FusionMeterProps) {
  const items = [
    { label: 'CAG',   value: weights.cag, icon: '💾' },
    { label: 'Graph', value: weights.graph, icon: '🕸️' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      className="fixed top-32 left-8 w-96"
    >
      <div className="bg-gray-900/95 backdrop-blur-2xl border border-gray-800 rounded-3xl p-10 shadow-2xl">
        {/* Header using CSS var accent — for a UI that doesn't look like a candy explosion */}
        <h2 className="text-3xl font-black text-[var(--accent)] mb-10 tracking-wider">FUSION ENGINE</h2>

        <div className="space-y-10">
          {items.map((item) => (
            <div key={item.label}>
              <div className="flex justify-between items-baseline mb-4">
                {/* Label with icon */}
                <span className="text-gray-300 font-medium text-lg flex items-center gap-2">
                  <span className="text-xl">{item.icon}</span>
                  {item.label}
                </span>
                <motion.span
                  key={item.value}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-white font-bold text-2xl font-mono"
                >
                  {(item.value * 100).toFixed(1)}%
                </motion.span>
              </div>

              {/* Rainbow finally meets its quiet end */}
              <div className="h-16 bg-gray-800/50 rounded-2xl overflow-hidden relative border border-gray-700">
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/50 to-[var(--accent)]/10 shadow-2xl"
                  initial={{ width: 0 }}
                  animate={{ width: `${item.value * 100}%` }}
                  transition={{ duration: 1.2, ease: "easeOut" }}
                >
                  <div className="absolute inset-0 bg-white/10 animate-pulse" />
                </motion.div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-10 pt-8 border-t border-gray-700 text-center">
          <div className="text-gray-500 text-sm font-light tracking-wider">
            RLFusion Orchestrator
          </div>
        </div>
      </div>
    </motion.div>
  );
}
