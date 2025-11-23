// Author: Bradley R. Kinnard


import { useEffect, useRef, useState } from 'react'
import './App.css'

interface Message {
  role: "user" | "assistant"
  content: string
  weights?: { rag: number; cag: number; graph: number }
  reward?: number
}

function App() {
  const [count, setCount] = useState(0)
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [input, setInput] = useState('')
  const wsRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current || isLoading) return

    setMessages(prev => [...prev, { role: 'user', content: input }])
    wsRef.current.send(JSON.stringify({ query: input, mode: 'chat' }))
    setInput('')
  }

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws')
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'start') {
        setIsLoading(true)
        setMessages(prev => [...prev, { role: 'assistant', content: '' }])
      } else if (data.type === 'token') {
        setMessages(prev => {
          const updated = [...prev]
          const lastMsg = updated[updated.length - 1]
          if (lastMsg && lastMsg.role === 'assistant') {
            lastMsg.content += data.token
          }
          return updated
        })
      } else if (data.type === 'done') {
        setIsLoading(false)
        setMessages(prev => {
          const updated = [...prev]
          const lastMsg = updated[updated.length - 1]
          if (lastMsg && lastMsg.role === 'assistant') {
            lastMsg.content = data.response
            lastMsg.weights = data.fusion_weights
            lastMsg.reward = data.reward
          }
          return updated
        })
      }
    }

    return () => {
      ws.close()
    }
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <div className="border-b border-white/10 p-4">
        <h1 className="text-2xl font-bold">RLFusion Orchestrator</h1>
        <p className="text-sm opacity-70">Local RL-driven AI • RTX 5070 • qwen2:7b</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-2xl rounded-lg p-4 ${
              msg.role === 'user' ? 'bg-primary/20' : 'bg-white/10'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
              {msg.weights && (
                <div className="text-sm mt-2 opacity-60">
                  RAG: {(msg.weights.rag * 100).toFixed(0)}% •
                  CAG: {(msg.weights.cag * 100).toFixed(0)}% •
                  Graph: {(msg.weights.graph * 100).toFixed(0)}%
                  {msg.reward !== undefined && ` • Reward: ${msg.reward.toFixed(2)}`}
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="text-center opacity-50">Thinking...</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-white/10 p-4">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask anything..."
            className="flex-1 bg-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-primary outline-none"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="bg-primary px-6 py-3 rounded-lg hover:bg-primary/80 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
