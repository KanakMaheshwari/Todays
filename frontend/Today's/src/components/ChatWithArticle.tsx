import React, { useState } from 'react';
import { Article } from '../data/articles';
import { SendIcon } from 'lucide-react';

type ChatWithArticleProps = {
  article: Article;
};

type Message = {
  id: string;
  sender: 'user' | 'ai';
  text: string;
};

export const ChatWithArticle = ({ article }: ChatWithArticleProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      sender: 'ai',
      text: `Hi! I'm your AI assistant. Ask me any questions about "${article.title}".`,
    },
  ]);
  const [inputValue, setInputValue] = useState('');

  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: inputValue,
    };
    setMessages((prev) => [...prev, userMessage]);
    const query = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/v1/chat/?query=${encodeURIComponent(query)}`,
        {
          method: 'POST',
        }
      );
      const aiResponseText = await response.text();
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: aiResponseText,
      };
      setMessages((prev) => [...prev, aiResponse]);
    } catch (error) {
      console.error('Error fetching chat response:', error);
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: 'Sorry, I am having trouble connecting to the server. Please try again later.',
      };
      setMessages((prev) => [...prev, aiResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-6 border rounded-lg shadow-sm">
      <div className="bg-blue-600 text-white p-4 rounded-t-lg flex justify-between items-center">
        <h3 className="font-medium">Chat with this article</h3>
      </div>
      <div className="h-80 overflow-y-auto p-4 bg-gray-50">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`mb-4 max-w-3/4 ${
              message.sender === 'user' ? 'ml-auto' : 'mr-auto'
            }`}
          >
            <div
              className={`p-3 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-blue-600 text-white rounded-br-none'
                  : 'bg-white border rounded-bl-none'
              }`}
            >
              {message.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="mb-4 max-w-3/4 mr-auto">
            <div className="p-3 rounded-lg bg-white border rounded-bl-none">
              <div className="flex items-center">
                <div className="dot-pulse"></div>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="p-4 border-t">
        <div className="flex">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask a question about this article..."
            className="flex-1 px-4 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            className="px-4 py-2 bg-blue-600 text-white rounded-r-lg hover:bg-blue-700"
            disabled={isLoading}
          >
            <SendIcon className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
};