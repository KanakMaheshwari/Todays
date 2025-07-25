import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { articles } from '../data/articles';
import { ArrowLeftIcon, MessageCircleIcon, ShareIcon, BookmarkIcon } from 'lucide-react';
import { ChatWithArticle } from './ChatWithArticle';
export const ArticlePage = () => {
  const {
    id
  } = useParams<{
    id: string;
  }>();
  const [showChat, setShowChat] = useState(false);
  const article = articles.find(a => a.id === id);
  if (!article) {
    return <div className="text-center py-12">
        <h2 className="text-2xl font-bold mb-4">Article not found</h2>
        <Link to="/" className="text-blue-600 hover:underline">
          Return to homepage
        </Link>
      </div>;
  }
  return <div className="max-w-3xl mx-auto">
      <Link to="/" className="inline-flex items-center text-blue-600 mb-6 hover:underline">
        <ArrowLeftIcon className="h-4 w-4 mr-1" />
        Back to all articles
      </Link>
      <h1 className="text-3xl md:text-4xl font-bold mb-4">{article.title}</h1>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <div className="h-10 w-10 rounded-full bg-gray-300 mr-3"></div>
          <div>
            <p className="font-medium">{article.author}</p>
            <p className="text-sm text-gray-500">{article.date}</p>
          </div>
        </div>
        <div className="flex space-x-2">
          <button className="p-2 rounded-full hover:bg-gray-100">
            <BookmarkIcon className="h-5 w-5 text-gray-600" />
          </button>
          <button className="p-2 rounded-full hover:bg-gray-100">
            <ShareIcon className="h-5 w-5 text-gray-600" />
          </button>
          <button className={`p-2 rounded-full ${showChat ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100 text-gray-600'}`} onClick={() => setShowChat(!showChat)}>
            <MessageCircleIcon className="h-5 w-5" />
          </button>
        </div>
      </div>
      <div className="mb-8">
        <img src={article.imageUrl} alt={article.title} className="w-full h-64 md:h-96 object-cover rounded-lg mb-4" />
        <p className="text-sm text-gray-500 italic">{article.imageCaption}</p>
      </div>
      <div className="prose max-w-none mb-8">
        {article.content.split('\n\n').map((paragraph, index) => <p key={index} className="mb-4">
            {paragraph}
          </p>)}
      </div>
      <div className="border-t pt-6">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-gray-500">Category: </span>
            <Link to="/" className="text-blue-600 capitalize hover:underline">
              {article.category}
            </Link>
          </div>
          <button onClick={() => setShowChat(!showChat)} className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <MessageCircleIcon className="h-5 w-5 mr-2" />
            Chat with this article
          </button>
        </div>
      </div>
      {showChat && <ChatWithArticle article={article} />}
    </div>;
};