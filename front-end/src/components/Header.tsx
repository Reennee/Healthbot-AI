"use client";

import { User, Heart } from "react-feather";

export default function Header() {
  return (
    <header className="bg-indigo-600 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Heart className="w-8 h-8" />
            <h1 className="text-2xl font-bold">HealthBot AI</h1>
          </div>
          <button className="bg-white text-indigo-600 px-4 py-2 rounded-full font-medium hover:bg-indigo-100 transition">
            <User className="inline mr-1" /> Sign In
          </button>
        </div>
        <p className="mt-2 opacity-90">Your virtual healthcare assistant powered by AI</p>
      </div>
    </header>
  );
}


