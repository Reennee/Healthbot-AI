"use client";

import { MessageCircle, ArrowRight, Heart, Shield, Zap } from "react-feather";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-4xl w-full text-center space-y-8">
        {/* Hero Section */}
        <div className="space-y-6">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Heart className="w-12 h-12 text-indigo-400" />
            <h1 className="text-5xl md:text-6xl font-bold text-white">HealthBot AI</h1>
          </div>
          <p className="text-xl md:text-2xl text-gray-300">
            Your intelligent healthcare assistant powered by AI
          </p>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            Get instant, reliable health information and guidance. Our AI assistant is trained to provide 
            accurate medical insights while always reminding you to consult healthcare professionals for serious concerns.
          </p>
        </div>

        {/* CTA Button */}
        <div className="pt-4">
          <Link
            href="/chat"
            className="inline-flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-8 py-4 rounded-full transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            <MessageCircle className="w-5 h-5" />
            <span>Start Chatting</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 p-6 rounded-xl hover:border-indigo-500 transition">
            <div className="bg-indigo-500/20 w-12 h-12 rounded-full flex items-center justify-center mb-4 mx-auto">
              <Zap className="text-indigo-400 w-6 h-6" />
            </div>
            <h3 className="font-bold text-lg mb-2 text-white">Instant Responses</h3>
            <p className="text-gray-400 text-sm">
              Get immediate answers to your health questions 24/7
            </p>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 p-6 rounded-xl hover:border-indigo-500 transition">
            <div className="bg-indigo-500/20 w-12 h-12 rounded-full flex items-center justify-center mb-4 mx-auto">
              <Shield className="text-indigo-400 w-6 h-6" />
            </div>
            <h3 className="font-bold text-lg mb-2 text-white">Privacy First</h3>
            <p className="text-gray-400 text-sm">
              Your conversations are secure and confidential
            </p>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 p-6 rounded-xl hover:border-indigo-500 transition">
            <div className="bg-indigo-500/20 w-12 h-12 rounded-full flex items-center justify-center mb-4 mx-auto">
              <Heart className="text-indigo-400 w-6 h-6" />
            </div>
            <h3 className="font-bold text-lg mb-2 text-white">Medical Accuracy</h3>
            <p className="text-gray-400 text-sm">
              Trained on reliable medical sources and best practices
            </p>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-12 p-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg">
          <p className="text-sm text-yellow-200">
            ⚠️ <strong>Important:</strong> HealthBot AI is not a substitute for professional medical advice, 
            diagnosis, or treatment. Always seek the advice of qualified health providers with any questions 
            you may have regarding a medical condition.
          </p>
        </div>
      </div>
    </div>
  );
}

