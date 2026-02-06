import Message from './Message'
import './MessageList.css'

export default function MessageList({ messages, lastMessageRef }) {
  return (
    <div className="message-list flex-grow-1 overflow-y-auto p-3 d-flex flex-column gap-2">
      {messages.map((message, index) => (
        <div
          key={message.id}
          ref={index === messages.length - 1 ? lastMessageRef : null}
        >
          <Message message={message} />
        </div>
      ))}
    </div>
  )
}
