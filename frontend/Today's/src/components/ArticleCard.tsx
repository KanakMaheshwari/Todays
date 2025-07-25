import React from 'react';
import { Article } from '../data/articles';
import { ClockIcon } from 'lucide-react';

type ArticleCardProps = {
  article: Article;
};

export const ArticleCard = ({ article }: ArticleCardProps) => {
  console.log(article);
  return (
    <div className="relative group">
      <a
        href={article.link}
        target="_blank"
        rel="noopener noreferrer"
        className="block bg-white rounded-lg overflow-hidden shadow-md transition-transform duration-300 hover:shadow-lg hover:-translate-y-1"
      >
        <div className="relative h-36">
          <img
            src={article.img_link}
            alt={article.title}
            className="w-full h-full object-cover"
          />
          <div className="absolute top-0 right-0 bg-blue-600 text-white px-2 py-1 text-xs font-medium capitalize">
            {article.category}
          </div>
        </div>
        <div className="p-4">
          <h2 className="text-lg font-bold mb-2 line-clamp-2">
            {article.title}
          </h2>

          <p className="text-gray-700 text-sm mb-3 line-clamp-4">
            {article.summary}
          </p>

          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>{article.author}</span>
            <div className="flex items-center">
              <ClockIcon className="h-4 w-4 mr-1" />
              <span>{article.date}</span>
            </div>
          </div>
        </div>
      </a>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-full max-w-sm p-3 bg-gray-800 text-white text-sm rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-50">
        <p>{article.summary}</p>
        <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-x-8 border-x-transparent border-t-8 border-t-gray-800"></div>
      </div>
    </div>
  );
};
