import React, { useState, useEffect } from 'react';
import { CategoryFilter } from './CategoryFilter';
import { ArticleList } from './ArticleList';
import { Article } from '../data/articles';

type HomePageProps = {
  searchQuery: string;
};

export const HomePage = ({ searchQuery }: HomePageProps) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [articles, setArticles] = useState<Article[]>([]);

  useEffect(() => {
    const fetchArticles = async () => {
      try {
        let url = 'http://127.0.0.1:8000/api/v1/articles/';
        if (selectedCategory) {
          url = `http://127.0.0.1:8000/api/v1/articles/category/${selectedCategory}`;
        } else if (searchQuery) {
          url = `http://127.0.0.1:8000/api/v1/articles/search/${searchQuery}`;
        }
        const response = await fetch(url);
        const data = await response.json();
        setArticles(data);
      } catch (error) {
        console.error('Error fetching articles:', error);
      }
    };

    fetchArticles();
  }, [selectedCategory, searchQuery]);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Latest News</h1>
      <CategoryFilter selectedCategory={selectedCategory} onSelectCategory={setSelectedCategory} />
      <ArticleList articles={articles} />
    </div>
  );
};