import { Menu } from 'lucide-react'

export default function Navbar({ sidebarOpen, onToggleSidebar }) {
  return (
    <nav className="navbar navbar-expand-lg navbar-light bg-light border-bottom">
      <div className="container-fluid px-3">
        <button
          className="btn btn-light"
          onClick={onToggleSidebar}
          title={sidebarOpen ? 'Fermer la barre latérale' : 'Ouvrir la barre latérale'}
        >
          <Menu size={24} />
        </button>
        <img src="/eilco.png" alt="Logo" style={{ height: '32px', width: '32px', objectFit: 'contain' }} className="ms-2" />
        <div className="flex-grow-1 text-center">
          <h1 className="h5 mb-0">Interface de Chat</h1>
        </div>
      </div>
    </nav>
  )
}
