"use client";

import VantaBackground from "../components/VantaBackground";
import Header from "../components/Header";
import HomePage from "../components/HomePage";
import Footer from "../components/Footer";

export default function Home() {
  return (
    <div id="vanta-bg" className="bg-gray-900 min-h-screen flex flex-col">
      <VantaBackground />
      <Header />
      <main className="grow">
        <HomePage />
      </main>
      <Footer />
    </div>
  );
}
