export default function Footer() {
  return (
    <footer className="bg-gray-900 border-t border-gray-800 text-white py-8">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 text-indigo-400">❤️</span>
              <span className="font-bold">HealthBot AI</span>
            </div>
            <p className="text-sm mt-2 text-gray-400">Your virtual healthcare companion</p>
          </div>
          <div className="flex space-x-6">
            <a href="#" className="hover:text-indigo-400 transition text-gray-300">Privacy</a>
            <a href="#" className="hover:text-indigo-400 transition text-gray-300">Terms</a>
            <a href="#" className="hover:text-indigo-400 transition text-gray-300">Contact</a>
          </div>
        </div>
        <div className="mt-6 pt-6 border-t border-gray-800 text-center text-sm text-gray-500">
          <p>© 2025 HealthBot AI. This is not a substitute for professional medical advice.</p>
        </div>
      </div>
    </footer>
  );
}


