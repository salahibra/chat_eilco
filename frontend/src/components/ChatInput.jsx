import { useState } from 'react'
import { Send } from 'lucide-react'
import './ChatInput.css'

export default function ChatInput({ onSendMessage }) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    setIsLoading(true)
    const message = input.trim()
    setInput('')

    try {
      onSendMessage(message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form className="chat-input-form border-top p-3 bg-white" onSubmit={handleSubmit}>
      <div className="d-flex gap-2 align-items-end">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Tapez votre question..."
          className="form-control"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="btn btn-primary flex-shrink-0"
          disabled={!input.trim() || isLoading}
          title="Envoyer le message"
        >
          <Send size={20} />
        </button>
      </div>
    </form>
  )
}
