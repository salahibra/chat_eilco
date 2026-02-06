import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import ChatContainer from './components/ChatContainer'
import './App.css'

// Generate a unique session ID
const generateSessionId = () => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

export default function App() {
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sessionId, setSessionId] = useState(null)

  // Initialize session ID on mount
  useEffect(() => {
    const storedSessionId = localStorage.getItem('chatSessionId')
    if (storedSessionId) {
      setSessionId(storedSessionId)
    } else {
      const newSessionId = generateSessionId()
      setSessionId(newSessionId)
      localStorage.setItem('chatSessionId', newSessionId)
    }
  }, [])

  const currentConversation = conversations.find(
    (conv) => conv.id === currentConversationId
  )

  const handleNewChat = () => {
    const newId = Date.now().toString()
    const newChatSessionId = generateSessionId()
    const newConversation = {
      id: newId,
      chatSessionId: newChatSessionId,
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
    }
    setConversations([newConversation, ...conversations])
    setCurrentConversationId(newId)
  }

  const handleSendMessage = async (message) => {
    if (!currentConversationId) {
      handleNewChat()
      return
    }

    // Add user message
    setConversations((prevConversations) =>
      prevConversations.map((conv) => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [
              ...conv.messages,
              { id: Date.now().toString(), text: message, sender: 'user' },
            ],
            title: conv.title === 'New Chat' ? message.substring(0, 30) : conv.title,
          }
        }
        return conv
      })
    )

    // Send message to API and get response
    try {
      console.log('Sending request to: /api/chat')
      const response = await fetch(`/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          session_id: currentConversation.chatSessionId || sessionId,
          message: message 
        }),
      })

      console.log('Response status:', response.status, response.statusText)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response body:', errorText)
        throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`)
      }

      const responseText = await response.text()
      console.log('Raw response text:', responseText)
      
      let data
      try {
        data = JSON.parse(responseText)
      } catch (parseError) {
        console.error('JSON parse error:', parseError)
        console.error('Response was:', responseText)
        throw new Error(`Invalid JSON response: ${responseText.substring(0, 200)}`)
      }
      
      console.log('API Response:', data)
      
      // Try multiple possible response field names
      let botResponse = ''
      if (typeof data === 'string') {
        botResponse = data
      } else if (data.response) {
        botResponse = data.response
      } else if (data.message) {
        botResponse = data.message
      } else if (data.result) {
        botResponse = data.result
      } else if (data.text) {
        botResponse = data.text
      } else if (data.answer) {
        botResponse = data.answer
      } else {
        // If none of the above, stringify the entire response
        botResponse = JSON.stringify(data)
      }

      // Add bot response to conversation
      setConversations((prevConversations) =>
        prevConversations.map((conv) => {
          if (conv.id === currentConversationId) {
            return {
              ...conv,
              messages: [
                ...conv.messages,
                {
                  id: Date.now().toString(),
                  text: botResponse,
                  sender: 'bot',
                  sources: data.sources || [],
                },
              ],
            }
          }
          return conv
        })
      )
    } catch (error) {
      console.error('Error sending message:', error)
      console.error('Error type:', error.name)
      console.error('Error message:', error.message)
      // Add error message to conversation
      setConversations((prevConversations) =>
        prevConversations.map((conv) => {
          if (conv.id === currentConversationId) {
            return {
              ...conv,
              messages: [
                ...conv.messages,
                {
                  id: Date.now().toString(),
                  text: `Erreur: ${error.message}`,
                  sender: 'bot',
                  sources: [],
                },
              ],
            }
          }
          return conv
        })
      )
    }
  }

  const handleDeleteConversation = (conversationId) => {
    const updatedConversations = conversations.filter(
      (conv) => conv.id !== conversationId
    )
    setConversations(updatedConversations)
    if (currentConversationId === conversationId) {
      setCurrentConversationId(
        updatedConversations.length > 0 ? updatedConversations[0].id : null
      )
    }
  }

  const handleSelectConversation = (conversationId) => {
    setCurrentConversationId(conversationId)
  }

  return (
    <div className="app d-flex flex-column vh-100">
      <Navbar sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
      <div className="app-container d-flex flex-grow-1 overflow-hidden">
        <Sidebar
          conversations={conversations}
          currentConversationId={currentConversationId}
          onNewChat={handleNewChat}
          onSelectConversation={handleSelectConversation}
          onDeleteConversation={handleDeleteConversation}
          isOpen={sidebarOpen}
        />
        <ChatContainer
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
        />
      </div>
    </div>
  )
}
