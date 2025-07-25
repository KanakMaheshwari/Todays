import React from 'react';

type SearchProps = {
  onSearch: (query: string) => void;
};

export const Search = ({ onSearch }: SearchProps) => {
  const [query, setQuery] = React.useState('');

  const handleSearch = () => {
    onSearch(query);
  };

  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="border border-gray-300 rounded-md px-4 py-2 w-full"
        placeholder="Search for articles..."
      />
      <button
        onClick={handleSearch}
        className="bg-blue-600 text-white px-4 py-2 rounded-md"
      >
        Search
      </button>
    </div>
  );
};