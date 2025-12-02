import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

export default function ConnectionStatus() {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');

  useEffect(() => {
    // Yeah, we're actually checking if the brain is home
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onopen = () => setStatus('connected');
    ws.onerror = ws.onclose = () => setStatus('error');
    return () => ws.close();
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-full max-w-sm"
    >
      {/* Outer glow container - this makes it pop off the dark background */}
      <div className="relative group">
        {/* The actual glass card */}
        <div className="relative bg-gray-900/40 backdrop-blur-2xl border border-gray-700/50 rounded-2xl px-6 py-4 shadow-2xl overflow-hidden">

          {/* Subtle gradient overlay for depth */}
          <div className="absolute inset-0 bg-gradient-to-br from-gray-800/20 to-transparent pointer-events-none" />

          {/* Content wrapper */}
          <div className="relative flex items-center gap-4">

            {/* Status indicator with pulsing glow */}
            <div className="relative flex-shrink-0">
              {/* Outer glow ring - only shows when connected */}
              {status === 'connected' && (
                <motion.div
                  animate={{
                    scale: [1, 1.5, 1],
                    opacity: [0.5, 0, 0.5],
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 2,
                    ease: "easeInOut"
                  }}
                  className="absolute inset-0 bg-emerald-500 rounded-full blur-md"
                />
              )}

              {/* The dot itself */}
              <motion.div
                animate={status === 'connected' ? {
                  scale: [1, 1.1, 1],
                } : {}}
                transition={{
                  repeat: Infinity,
                  duration: 2,
                  ease: "easeInOut"
                }}
                className={`relative w-3 h-3 rounded-full ${
                  status === 'connected'
                    ? 'bg-emerald-400 shadow-lg shadow-emerald-500/60'
                    : status === 'error'
                    ? 'bg-red-500 shadow-lg shadow-red-500/60'
                    : 'bg-amber-400 shadow-lg shadow-amber-500/60'
                }`}
              />
            </div>

            {/* Text content */}
            <div className="flex-1 min-w-0">
              {/* Status text */}
              <motion.div
                className={`font-semibold text-sm tracking-wide mb-0.5 ${
                  status === 'connected'
                    ? 'text-emerald-400'
                    : status === 'error'
                    ? 'text-red-400'
                    : 'text-amber-400'
                }`}
                animate={status === 'connecting' ? {
                  opacity: [0.5, 1, 0.5]
                } : {}}
                transition={{
                  repeat: Infinity,
                  duration: 1.5
                }}
              >
                {status === 'connected' ? 'RLFO • CONNECTED' :
                 status === 'error' ? 'BACKEND OFFLINE' : 'INITIALIZING...'}
              </motion.div>

              {/* GPU info - more subtle, but still readable */}
              <div className="text-gray-400 text-xs font-light tracking-wide">
                RTX 5070 • Blackwell • {status === 'connected' ? '11 GB VRAM' : 'Waiting...'}
              </div>
            </div>

            {/* Subtle animated corner accent - because why not */}
            {status === 'connected' && (
              <motion.div
                animate={{
                  opacity: [0.3, 0.6, 0.3],
                }}
                transition={{
                  repeat: Infinity,
                  duration: 3,
                  ease: "easeInOut"
                }}
                className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-emerald-500/10 to-transparent rounded-2xl pointer-events-none"
              />
            )}
          </div>

          {/* Bottom border glow when connected */}
          {status === 'connected' && (
            <motion.div
              animate={{
                opacity: [0.3, 0.6, 0.3],
              }}
              transition={{
                repeat: Infinity,
                duration: 2.5,
                ease: "easeInOut"
              }}
              className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent"
            />
          )}
        </div>

        {/* Hover effect - subtle lift */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-br from-emerald-500/0 to-transparent rounded-2xl pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        />
      </div>
    </motion.div>
  );
}

// Usage: Just drop this wherever you need it in your layout
// It'll flow responsively and look gorgeous on your #0a0a0a background
