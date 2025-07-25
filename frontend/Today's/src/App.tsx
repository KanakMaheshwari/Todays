import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { HomePage } from './components/HomePage';
import { ArticlePage } from './components/ArticlePage';
import { FloatingChatButton } from './components/FloatingChatButton';
import { FloatingChatWindow } from './components/FloatingChatWindow';

export function App() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen bg-gray-50">
        <Header onSearch={setSearchQuery} />
        <main className="flex-1 container mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<HomePage searchQuery={searchQuery} />} />
            <Route path="/article/:id" element={<ArticlePage />} />
          </Routes>
        </main>
        <footer className="bg-gray-800 text-white py-6">
          <div className="container mx-auto px-4">
            <p className="text-center">© Today's - by kanak</p>
          </div>
        </footer>
        <FloatingChatButton onClick={toggleChat} />
        {isChatOpen && <FloatingChatWindow onClose={toggleChat} />}
      </div>
    </BrowserRouter>
  );
}