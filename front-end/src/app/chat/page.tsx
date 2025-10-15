"use client";

import VantaBackground from "../../components/VantaBackground";
import Header from "../../components/Header";
import Chat from "../../components/Chat";
import Features from "../../components/Features";
import Footer from "../../components/Footer";

export default function ChatPage() {
  return (
    <div id="vanta-bg" className="bg-gray-900 min-h-screen flex flex-col">
      <VantaBackground />
      <Header />
      <main className="grow container mx-auto px-4 py-8 flex flex-col">
        <div className="bg-gray-800 rounded-xl shadow-2xl overflow-hidden flex flex-col grow border border-gray-700">
          <Chat />
        </div>
        <Features />
      </main>
      <Footer />
    </div>
  );
}

