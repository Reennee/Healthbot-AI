import { Activity, BookOpen, MapPin } from "react-feather";

export default function Features() {
  return (
    <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition">
        <div className="bg-indigo-100 w-12 h-12 rounded-full flex items-center justify-center mb-4">
          <Activity className="text-indigo-600 w-5 h-5" />
        </div>
        <h3 className="font-bold text-lg mb-2 text-black">Symptom Checker</h3>
        <p className="text-gray-600">Get preliminary assessment of your symptoms based on medical knowledge.</p>
      </div>

      <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition">
        <div className="bg-indigo-100 w-12 h-12 rounded-full flex items-center justify-center mb-4">
          <BookOpen className="text-indigo-600 w-5 h-5" />
        </div>
        <h3 className="font-bold text-lg mb-2 text-black">Health Education</h3>
        <p className="text-gray-600">Learn about medical conditions, treatments, and healthy lifestyle choices.</p>
      </div>

      <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition">
        <div className="bg-indigo-100 w-12 h-12 rounded-full flex items-center justify-center mb-4">
          <MapPin className="text-indigo-600 w-5 h-5" />
        </div>
        <h3 className="font-bold text-lg mb-2 text-black">Find Care</h3>
        <p className="text-gray-600">Get recommendations for nearby healthcare providers when needed.</p>
      </div>
    </div>
  );
}


