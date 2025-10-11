"use client";

import { useEffect, useRef, useState } from "react";
import { MessageSquare, Send } from "react-feather";
import { healthBotAPI, ChatResponse } from "../lib/api";

type ChatMessage = {
  id: string;
  role: "user" | "bot";
  text: string;
  timestamp: string;
  confidence?: number;
  domain_relevance?: boolean;
};

function nowLabel(): string {
  return "Just now";
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "bot",
      text: "Hello! I'm your HealthBot AI assistant. How can I help you today?",
      timestamp: nowLabel(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId, setConversationId] = useState<string>("");
  const [apiError, setApiError] = useState<string>("");
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  async function handleSend() {
    const trimmed = input.trim();
    if (!trimmed) return;
    
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text: trimmed,
      timestamp: nowLabel(),
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setApiError("");
    setIsTyping(true);
    
    try {
      const response: ChatResponse = await healthBotAPI.chat(trimmed, conversationId || undefined);
      
      const botMsg: ChatMessage = {
        id: `bot-${Date.now()}`,
        role: "bot",
        text: response.response,
        timestamp: nowLabel(),
        confidence: response.confidence,
        domain_relevance: response.domain_relevance,
      };
      
      setMessages((m) => [...m, botMsg]);
      setConversationId(response.conversation_id);
    } catch (error) {
      console.error("API Error:", error);
      setApiError("Sorry, I'm having trouble connecting to the AI service. Please try again.");
      
      // Fallback response
      const fallbackMsg: ChatMessage = {
        id: `bot-${Date.now()}`,
        role: "bot",
        text: "I'm sorry, I'm having trouble processing your request right now. Please try again or contact support if the issue persists.",
        timestamp: nowLabel(),
      };
      setMessages((m) => [...m, fallbackMsg]);
    } finally {
      setIsTyping(false);
    }
  }

  return (
    <>
      {/* Chat Header */}
      <div className="bg-indigo-500 text-white p-4 flex items-center">
        <div className="w-10 h-10 rounded-full bg-indigo-400 flex items-center justify-center">
          <MessageSquare className="w-5 h-5" />
        </div>
        <div className="ml-3">
          <h2 className="font-bold">HealthBot Assistant</h2>
          <p className="text-xs opacity-80">Online</p>
        </div>
      </div>

      {/* Chat Messages */}
      <div ref={containerRef} className="chat-container chat-container-scroll p-4 space-y-3 flex-grow bg-gray-50">
        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
            <div className={m.role === "user" ? "message-bubble bg-indigo-500 text-white p-3 shadow-md max-w-[80%] rounded-[1rem]" : "message-bubble bg-white text-black p-3 shadow-md max-w-[80%] rounded-[1rem]"}>
              <p>{m.text}</p>
              <p className={m.role === "user" ? "text-xs text-indigo-100 mt-1" : "text-xs text-gray-500 mt-1"}>{m.timestamp}</p>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start">
            <div className="message-bubble bg-white  p-3 shadow-md max-w-[80%] rounded-[1rem]">
              <div className="typing-indicator flex items-center space-x-1">
                <span className="typing-dot inline-block w-2 h-2 rounded-full bg-indigo-600 mx-[2px] animate-bounce" />
                <span className="typing-dot inline-block w-2 h-2 rounded-full bg-indigo-600 mx-[2px] animate-bounce" />
                <span className="typing-dot inline-block w-2 h-2 rounded-full bg-indigo-600 mx-[2px] animate-bounce" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4 bg-white text-black">
        <div className="flex space-x-2">
          <input
            type="text"
            placeholder="Type your health concern..."
            className="flex-grow border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
          />
          <button title="Send" className="bg-indigo-500 text-white rounded-full p-2 hover:bg-indigo-600 transition" onClick={handleSend}>
            <Send className="w-5 h-5" />
          </button>
        </div>
        <div className="mt-2 flex justify-center text-xs text-gray-600">
          <span>Not a substitute for professional medical advice</span>
        </div>
        {apiError && (
          <div className="mt-2 p-2 bg-red-100 border border-red-300 rounded text-xs text-red-700">
            {apiError}
          </div>
        )}
      </div>
    </>
  );
}


