import { useEffect, useRef } from 'react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'

export default function ChatContainer({ conversation, onSendMessage }) {
  const messagesEndRef = useRef(null)
  const lastMessageRef = useRef(null)

  const scrollToNewMessage = () => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  useEffect(() => {
    if (conversation?.messages && conversation.messages.length > 0) {
      const lastMessage = conversation.messages[conversation.messages.length - 1]
      // Scroll to beginning of message if it's from bot
      if (lastMessage.sender !== 'user') {
        scrollToNewMessage()
      } else {
        // Scroll to bottom if it's a user message
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }, [conversation?.messages])

  if (!conversation) {
    return (
      <div className="chat-container d-flex align-items-center justify-content-center flex-grow-1 bg-light">
        <div className="text-center d-flex flex-column align-items-center gap-3">
          <img src="/eilco.png" alt="Logo" style={{ height: '80px', width: '80px', objectFit: 'contain', opacity: 0.5 }} />
          <h2 className="h4">Bienvenue dans Interface de Chat</h2>
          <p className="text-muted">Commencez une nouvelle conversation ou sélectionnez-en une dans l'historique</p>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-container d-flex flex-column flex-grow-1 bg-white">
      <MessageList messages={conversation.messages} lastMessageRef={lastMessageRef} />
      <div ref={messagesEndRef} />
      <ChatInput onSendMessage={onSendMessage} />
    </div>
  )
}
