import React from 'react';
type CategoryFilterProps = {
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
};
export const CategoryFilter = ({
  selectedCategory,
  onSelectCategory
}: CategoryFilterProps) => {
  const categories = ['finance', 'sports', 'lifestyle','music'];
  return <div className="flex flex-wrap gap-2">
      <button onClick={() => onSelectCategory(null)} className={`px-4 py-2 rounded-full text-sm font-medium ${selectedCategory === null ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'}`}>
        All
      </button>
      {categories.map(category => <button key={category} onClick={() => onSelectCategory(category)} className={`px-4 py-2 rounded-full text-sm font-medium capitalize ${selectedCategory === category ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'}`}>
          {category}
        </button>)}
    </div>;
};