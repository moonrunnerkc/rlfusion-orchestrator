// Author: Bradley R. Kinnard
// reindex-button: upload + auto-reindex sub-panel for the right sidebar.

import { useRef, useState, type DragEvent, type ChangeEvent, type KeyboardEvent } from 'react';

type Status = 'idle' | 'uploading' | 'indexing' | 'done' | 'empty' | 'error';

const ALLOWED = ['.txt', '.md', '.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'];

export function ReindexButton() {
  const [status, setStatus] = useState<Status>('idle');
  const [details, setDetails] = useState<string>('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetSoon = (ms: number) => {
    setTimeout(() => {
      setStatus('idle');
      setDetails('');
    }, ms);
  };

  const uploadAndIndex = async (files: FileList | File[]) => {
    const valid = Array.from(files).filter((f) =>
      ALLOWED.some((ext) => f.name.toLowerCase().endsWith(ext)),
    );
    if (valid.length === 0) {
      setStatus('error');
      setDetails('Only .txt, .md, .pdf, and image files accepted');
      resetSoon(3000);
      return;
    }

    setStatus('uploading');
    setDetails(`Uploading ${valid.length} file${valid.length > 1 ? 's' : ''}...`);
    const form = new FormData();
    valid.forEach((f) => form.append('files', f));

    try {
      const upRes = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: form,
      });
      const upData = await upRes.json();
      if (!upRes.ok || upData.total_saved === 0) {
        setStatus('error');
        setDetails(upData.message || upData.detail || 'Upload failed');
        resetSoon(3000);
        return;
      }

      setStatus('indexing');
      setDetails(`Uploaded ${upData.total_saved}, building index...`);
      const ixRes = await fetch('http://localhost:8000/api/reindex', { method: 'POST' });
      const ixData = await ixRes.json();
      setStatus('done');
      const imgNote = ixData.images_indexed > 0 ? `, ${ixData.images_indexed} images` : '';
      setDetails(
        `${ixData.files_processed} files, ${ixData.chunks_indexed} chunks${imgNote} (${ixData.elapsed_seconds}s)`,
      );
      resetSoon(6000);
    } catch {
      setStatus('error');
      setDetails('Backend unreachable');
      resetSoon(4000);
    }
  };

  const handleReindexOnly = async () => {
    setStatus('indexing');
    setDetails('Rebuilding index...');
    try {
      const res = await fetch('http://localhost:8000/api/reindex', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'empty') {
        setStatus('empty');
        setDetails('No documents in data/docs/');
      } else {
        setStatus('done');
        const imgNote = data.images_indexed > 0 ? `, ${data.images_indexed} images` : '';
        setDetails(
          `${data.files_processed} files, ${data.chunks_indexed} chunks${imgNote} (${data.elapsed_seconds}s)`,
        );
      }
      resetSoon(5000);
    } catch {
      setStatus('error');
      setDetails('Backend unreachable');
      resetSoon(4000);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) uploadAndIndex(e.dataTransfer.files);
  };

  const onFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadAndIndex(e.target.files);
      e.target.value = '';
    }
  };

  const onDropZoneKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const busy = status === 'uploading' || status === 'indexing';

  return (
    <div className="pt-4 border-t border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true">📄</span>
          <span className="text-sm text-[var(--accent)]">RAG Documents</span>
        </div>
        <button
          type="button"
          onClick={handleReindexOnly}
          disabled={busy}
          className="text-xs text-gray-400 hover:text-cyan-400 transition disabled:opacity-40"
          title="Rebuild index without uploading"
        >
          ↻ Reindex
        </button>
      </div>

      <div
        role="button"
        tabIndex={0}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !busy && fileInputRef.current?.click()}
        onKeyDown={onDropZoneKeyDown}
        className={`w-full border-2 border-dashed rounded-lg px-3 py-4 text-center cursor-pointer transition ${
          dragOver
            ? 'border-cyan-400 bg-cyan-500/10'
            : busy
            ? 'border-gray-700 bg-gray-800/30 cursor-wait'
            : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/30'
        }`}
        aria-label="Upload documents to data/docs"
      >
        {busy ? (
          <p className="text-xs text-gray-400 animate-pulse">{details}</p>
        ) : (
          <>
            <p className="text-sm text-gray-400">Drop files here</p>
            <p className="text-xs text-gray-600 mt-1">.txt .md .pdf .png .jpg</p>
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.webp,.bmp,.tiff"
        onChange={onFileSelect}
        className="hidden"
      />

      {status === 'done' && details && <p className="text-xs text-emerald-400 mt-1">✓ {details}</p>}
      {status === 'empty' && details && <p className="text-xs text-amber-400 mt-1">⚠ {details}</p>}
      {status === 'error' && details && <p className="text-xs text-red-400 mt-1">✕ {details}</p>}
    </div>
  );
}
