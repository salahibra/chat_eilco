import { Plus, Trash2 } from 'lucide-react'
import './Sidebar.css'

export default function Sidebar({
  conversations,
  currentConversationId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  isOpen,
}) {
  return (
    <aside className={`sidebar flex-shrink-0 border-end bg-light ${isOpen ? 'open' : 'closed'}`} style={{ width: '260px' }}>
      <div className="border-bottom p-3">
        <button className="btn btn-outline-secondary w-100 d-flex align-items-center justify-content-center gap-2" onClick={onNewChat} title="Nouveau chat">
          <Plus size={20} />
          <span>Nouveau Chat</span>
        </button>
      </div>

      <div className="overflow-y-auto flex-grow-1 p-3">
        <div>
          <h6 className="text-muted text-uppercase ps-2 mb-2" style={{ fontSize: '11px', fontWeight: '700' }}>Historique des Conversations</h6>
          <div className="d-flex flex-column gap-1">
            {conversations.length === 0 ? (
              <p className="text-center text-muted small py-3 mb-0">Aucune conversation pour le moment</p>
            ) : (
              conversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`d-flex align-items-center justify-content-between p-2 rounded cursor-pointer transition-all ${
                    currentConversationId === conversation.id ? 'bg-secondary-subtle border' : 'border-0'
                  }`}
                  onClick={() => onSelectConversation(conversation.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="flex-grow-1 min-width-0">
                    <p className="small mb-0 text-truncate">{conversation.title}</p>
                    <span className="text-muted" style={{ fontSize: '12px' }}>
                      {new Date(conversation.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    className="btn btn-sm border-0 text-muted ms-2 delete-btn"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteConversation(conversation.id)
                    }}
                    title="Supprimer la conversation"
                    style={{ opacity: 0 }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}
