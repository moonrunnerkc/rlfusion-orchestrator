// frontend/src/components/SettingsPanel.tsx
// Author: Bradley R. Kinnard
// Settings panel with web browsing toggle

import { useEffect, useState } from 'react';

interface SettingsPanelProps {
  onClose: () => void;
}

export default function SettingsPanel({ onClose }: SettingsPanelProps) {
  const [webEnabled, setWebEnabled] = useState(true);
  const [loading, setLoading] = useState(false);

  // Load current config on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/config')
      .then(res => res.json())
      .then(data => setWebEnabled(data.web?.enabled ?? true))
      .catch(err => console.error('Failed to load config:', err));
  }, []);

  const toggleWeb = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ web: { enabled: !webEnabled } })
      });

      if (response.ok) {
        setWebEnabled(!webEnabled);
      }
    } catch (err) {
      console.error('Failed to update config:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 max-w-md w-full shadow-2xl">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Web Toggle */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🌐</span>
              <label className="text-lg font-medium text-white">Live Web Browsing</label>
            </div>
            <button
              onClick={toggleWeb}
              disabled={loading}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                webEnabled ? 'bg-[var(--accent)]' : 'bg-gray-700'
              } ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  webEnabled ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <p className="text-sm text-gray-400 ml-7">
            When off, 100% offline – no internet used ever
          </p>
        </div>
      </div>
    </div>
  );
}
