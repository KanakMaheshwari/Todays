import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { SearchIcon, BellIcon, UserIcon } from 'lucide-react';

type HeaderProps = {
  onSearch: (query: string) => void;
};

export const Header = ({ onSearch }: HeaderProps) => {
  const [inputValue, setInputValue] = useState('');

  const handleSearch = () => {
    onSearch(inputValue);
  };

  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <Link to="/" className="text-2xl font-bold text-blue-600">
            Today's
          </Link>
          <div className="flex-1 flex items-center justify-center">
            <div className="relative flex items-center">
              <input
                type="text"
                placeholder="Search articles..."
                className="w-80 pl-9 pr-3 py-1 border border-gray-300 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button onClick={handleSearch} className="absolute left-2 top-1/2 transform -translate-y-1/2 text-gray-400">
                <SearchIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};