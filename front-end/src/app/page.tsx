"use client";

import VantaBackground from "../components/VantaBackground";
import Header from "../components/Header";
import Chat from "../components/Chat";
import Features from "../components/Features";
import Footer from "../components/Footer";

export default function Home() {
  return (
    <div id="vanta-bg" className="bg-gray-50 min-h-screen flex flex-col">
      <VantaBackground />
      <Header />
      <main className="flex-grow container mx-auto px-4 py-8 flex flex-col">
        <div className="bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col flex-grow">
          <Chat />
        </div>
        <Features />
      </main>
      <Footer />
    </div>
  );
}
