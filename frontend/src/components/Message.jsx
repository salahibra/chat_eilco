import { useState } from 'react'
import { Copy, Check } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './Message.css'

export default function Message({ message }) {
  const [copied, setCopied] = useState(false)
  const [hoveredSourceIndex, setHoveredSourceIndex] = useState(null)
  const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0, showAbove: true })

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleSourceMouseEnter = (e, idx) => {
    setHoveredSourceIndex(idx)
    const rect = e.currentTarget.getBoundingClientRect()
    const tooltipHeight = 250 // Approximate height of tooltip
    const spaceAbove = rect.top
    const spaceBelow = window.innerHeight - rect.bottom
    
    // If more space below than above, show below; otherwise show above
    const showAbove = spaceAbove > spaceBelow && spaceAbove > tooltipHeight + 20

    setTooltipPos({
      top: showAbove ? rect.top : rect.bottom,
      left: rect.left + rect.width / 2,
      showAbove,
    })
  }

  const handleSourceMouseLeave = () => {
    setHoveredSourceIndex(null)
  }

  const isUser = message.sender === 'user'
  const sources = message.sources || []

  return (
    <div className={`d-flex gap-2 message-wrapper ${isUser ? 'justify-content-end' : 'justify-content-start'}`}>
      <div className={`message-content rounded-3 p-3 ${isUser ? 'bg-primary text-white' : 'bg-light text-dark'}`} style={{ maxWidth: '70%' }}>
        {isUser ? (
          <p className="mb-0">{message.text}</p>
        ) : (
          <div className="bot-message-text">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
            >
              {message.text}
            </ReactMarkdown>
            
            {/* Sources at the end as [1], [2], ... */}
            {sources.length > 0 && (
              <div className="sources-footer">
                {sources.map((source, idx) => (
                  <span
                    key={idx}
                    className="source-reference"
                    onMouseEnter={(e) => handleSourceMouseEnter(e, idx)}
                    onMouseLeave={handleSourceMouseLeave}
                  >
                    [{idx + 1}]
                  </span>
                ))}
              </div>
            )}
            
            {/* Source tooltip on hover */}
            {hoveredSourceIndex !== null && sources[hoveredSourceIndex] && (
              <div 
                className="source-tooltip"
                style={{
                  top: `${tooltipPos.top}px`,
                  left: `${tooltipPos.left}px`,
                  transform: tooltipPos.showAbove 
                    ? 'translate(-50%, -100%)' 
                    : 'translate(-50%, 8px)',
                }}
              >
                <div className="source-header">
                  <strong>{sources[hoveredSourceIndex].Filename}</strong>
                  <span className="source-page">Page {sources[hoveredSourceIndex].Page}</span>
                </div>
                <div className="source-content">
                  {sources[hoveredSourceIndex].Content}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      {!isUser && (
        <button
          className="btn btn-sm border-0 align-self-end copy-btn"
          onClick={handleCopy}
          title="Copier le message"
          style={{ opacity: 0 }}
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
        </button>
      )}
    </div>
  )
}
