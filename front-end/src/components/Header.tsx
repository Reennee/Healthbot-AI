"use client";

import { User, Heart } from "react-feather";
import Link from "next/link";

export default function Header() {
  return (
    <header className="bg-gray-800 border-b border-gray-700 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition">
            <Heart className="w-8 h-8 text-indigo-400" />
            <h1 className="text-2xl font-bold">HealthBot AI</h1>
          </Link>
          <button className="bg-indigo-600 text-white px-4 py-2 rounded-full font-medium hover:bg-indigo-700 transition">
            <User className="inline mr-1" /> Sign In
          </button>
        </div>
        <p className="mt-2 opacity-90 text-gray-300">Your virtual healthcare assistant powered by AI</p>
      </div>
    </header>
  );
}


