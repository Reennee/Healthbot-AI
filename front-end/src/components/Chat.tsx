"use client";

import { useEffect, useRef, useState } from "react";
import { MessageSquare, Send, Download, Trash2, BarChart, AlertTriangle } from "react-feather";
import { healthBotAPI } from "../lib/api";

type ChatMessage = {
  id: string;
  role: "user" | "bot";
  text: string;
  timestamp: string;
  confidence?: number;
  domain_relevance?: boolean;
  intent?: string;
  quality_metrics?: {
    response_length: number;
    has_disclaimer: boolean;
    has_emergency_guidance: boolean;
    response_relevance: number;
  };
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
  const [showMetrics, setShowMetrics] = useState(false);
  type QualityMetrics = {
    avg_response_length?: number;
    avg_length_ratio?: number;
    disclaimer_rate?: number;
    emergency_guidance_rate?: number;
    avg_relevance?: number;
  };
  type EvaluationMetrics = {
    bleu_score?: number;
    rouge_l?: number;
    perplexity?: number;
    eval_loss?: number;
    quality_metrics?: QualityMetrics;
    total_samples?: number;
    successful_predictions?: number;
    evaluation_timestamp?: string;
  };
  type ModelPerformance = {
    model_info?: {
      model_name: string;
      model_type: string;
      is_fine_tuned: boolean;
      is_ready: boolean;
    };
    evaluation_metrics?: EvaluationMetrics;
    conversation_stats?: {
      total_conversations: number;
      total_messages: number;
    };
    healthcare_domain_coverage?: {
      keywords_covered: number;
      intent_categories: number;
    };
  };
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
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
      // Get response with quality analysis
      const response = await healthBotAPI.chatWithQuality(trimmed, conversationId || undefined);
      
      const botMsg: ChatMessage = {
        id: `bot-${Date.now()}`,
        role: "bot",
        text: response.response.response,
        timestamp: nowLabel(),
        confidence: response.response.confidence,
        domain_relevance: response.response.domain_relevance,
        quality_metrics: response.quality_metrics,
      };
      
      setMessages((m) => [...m, botMsg]);
      setConversationId(response.response.conversation_id);
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

  async function handleAnalyzePerformance() {
    setIsAnalyzing(true);
    try {
      const performance = await healthBotAPI.getModelPerformance();
      setModelPerformance(performance);
      setShowMetrics(true);
    } catch (error) {
      console.error("Error fetching performance metrics:", error);
      setApiError("Failed to load performance metrics");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function handleClearChat() {
    setMessages([
      {
        id: "welcome",
        role: "bot",
        text: "Hello! I'm your HealthBot AI assistant. How can I help you today?",
        timestamp: nowLabel(),
      },
    ]);
    setConversationId("");
    setModelPerformance(null);
    setShowMetrics(false);
  }

  function handleExportChat() {
    const chatData = {
      conversation_id: conversationId,
      messages: messages,
      export_timestamp: new Date().toISOString(),
      total_messages: messages.length
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `healthbot-chat-${conversationId || 'export'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <>
    <div className="flex flex-col h-[70vh]">
      {/* Chat Header */}
      <div className="bg-gray-800 border-b border-gray-700 text-white p-4 flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center">
            <MessageSquare className="w-5 h-5" />
          </div>
          <div className="ml-3">
            <h2 className="font-bold">HealthBot Assistant</h2>
            <p className="text-xs opacity-80">Online • Healthcare AI</p>
          </div>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={handleAnalyzePerformance}
            disabled={isAnalyzing}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 p-2 rounded-full transition"
            title="Analyze Performance"
          >
            <BarChart className="w-4 h-4" />
          </button>
          <button
            onClick={handleExportChat}
            className="bg-indigo-600 hover:bg-indigo-700 p-2 rounded-full transition"
            title="Export Chat"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={handleClearChat}
            className="bg-red-600 hover:bg-red-700 p-2 rounded-full transition"
            title="Clear Chat"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Chat Messages */}
      <div ref={containerRef} className="chat-container chat-container-scroll p-4 space-y-3 grow bg-gray-900">
        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
            <div className={m.role === "user" ? "message-bubble bg-indigo-600 text-white p-3 shadow-md max-w-[80%] rounded-2xl" : "message-bubble bg-gray-800 text-gray-100 p-3 shadow-md max-w-[80%] rounded-2xl border border-gray-700"}>
              <p>{m.text}</p>
              <div className="flex items-center justify-between mt-1">
                <p className={m.role === "user" ? "text-xs text-indigo-100" : "text-xs text-gray-400"}>{m.timestamp}</p>
                {m.role === "bot" && m.quality_metrics && (
                  <div className="flex items-center space-x-2 text-xs">
                    {m.quality_metrics.has_disclaimer && (
                      <span className="bg-green-900/50 text-green-300 px-2 py-1 rounded-full border border-green-700">Medical Disclaimer</span>
                    )}
                    {m.quality_metrics.has_emergency_guidance && (
                      <span className="bg-red-900/50 text-red-300 px-2 py-1 rounded-full flex items-center border border-red-700">
                        <AlertTriangle className="w-3 h-3 mr-1" />
                        Emergency
                      </span>
                    )}
                    {m.confidence && (
                      <span className="text-gray-400">
                        Confidence: {Math.round(m.confidence * 100)}%
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start">
            <div className="message-bubble bg-gray-800 border border-gray-700 p-3 shadow-md max-w-[80%] rounded-2xl">
              <div className="typing-indicator flex items-center space-x-1">
                <span className="typing-dot inline-block w-2 h-2 rounded-full bg-indigo-400 mx-[2px] animate-bounce" />
                <span className="typing-dot inline-block w-2 h-2 rounded-full bg-indigo-400 mx-[2px] animate-bounce" />
                <span className="typing-dot inline-block w-2 h-2 rounded-full bg-indigo-400 mx-[2px] animate-bounce" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-700 p-4 bg-gray-800 text-white">
        <div className="flex space-x-2">
          <input
            type="text"
            placeholder="Type your health concern..."
            className="grow border border-gray-600 bg-gray-700 text-white rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
          />
          <button title="Send" className="bg-indigo-600 text-white rounded-full p-2 hover:bg-indigo-700 transition" onClick={handleSend}>
            <Send className="w-5 h-5" />
          </button>
        </div>
        <div className="mt-2 flex justify-center text-xs text-gray-400">
          <span>Not a substitute for professional medical advice</span>
        </div>
        {apiError && (
          <div className="mt-2 p-2 bg-red-900/50 border border-red-700 rounded text-xs text-red-300">
            {apiError}
          </div>
        )}
      </div>
    </div>
      {/* Performance Metrics Modal */}
      {showMetrics && modelPerformance && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white">Model Performance Metrics</h3>
              <button
                onClick={() => setShowMetrics(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              {/* Model Info */}
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
                <h4 className="font-semibold mb-2 text-white">Model Information</h4>
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                  <div>
                    <span className="font-medium">Model:</span> {modelPerformance.model_info?.model_name}
                  </div>
                  <div>
                    <span className="font-medium">Type:</span> {modelPerformance.model_info?.model_type}
                  </div>
                  <div>
                    <span className="font-medium">Fine-tuned:</span> {modelPerformance.model_info?.is_fine_tuned ? 'Yes' : 'No'}
                  </div>
                  <div>
                    <span className="font-medium">Status:</span> {modelPerformance.model_info?.is_ready ? 'Ready' : 'Not Ready'}
                  </div>
                </div>
              </div>

              {/* Evaluation Metrics */}
              {modelPerformance.evaluation_metrics && (
                <div className="bg-blue-900/30 border border-blue-700/50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2 text-white">Evaluation Metrics</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                    <div>
                      <span className="font-medium">BLEU Score:</span> {modelPerformance.evaluation_metrics.bleu_score?.toFixed(3) || 'N/A'}
                    </div>
                    <div>
                      <span className="font-medium">ROUGE-L:</span> {modelPerformance.evaluation_metrics.rouge_l?.toFixed(3) || 'N/A'}
                    </div>
                    <div>
                      <span className="font-medium">Total Samples:</span> {modelPerformance.evaluation_metrics.total_samples || 0}
                    </div>
                    <div>
                      <span className="font-medium">Successful Predictions:</span> {modelPerformance.evaluation_metrics.successful_predictions || 0}
                    </div>
                  </div>
                </div>
              )}

              {/* Quality Metrics */}
              {modelPerformance.evaluation_metrics?.quality_metrics && (
                <div className="bg-green-900/30 border border-green-700/50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2 text-white">Quality Metrics</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                    <div>
                      <span className="font-medium">Avg Response Length:</span> {Math.round(modelPerformance.evaluation_metrics.quality_metrics.avg_response_length || 0)}
                    </div>
                    <div>
                      <span className="font-medium">Disclaimer Rate:</span> {Math.round((modelPerformance.evaluation_metrics.quality_metrics.disclaimer_rate || 0) * 100)}%
                    </div>
                    <div>
                      <span className="font-medium">Emergency Guidance Rate:</span> {Math.round((modelPerformance.evaluation_metrics.quality_metrics.emergency_guidance_rate || 0) * 100)}%
                    </div>
                    <div>
                      <span className="font-medium">Avg Relevance:</span> {Math.round((modelPerformance.evaluation_metrics.quality_metrics.avg_relevance || 0) * 100)}%
                    </div>
                  </div>
                </div>
              )}

              {/* Conversation Stats */}
              <div className="bg-purple-900/30 border border-purple-700/50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2 text-white">Conversation Statistics</h4>
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                  <div>
                    <span className="font-medium">Total Conversations:</span> {modelPerformance.conversation_stats?.total_conversations || 0}
                  </div>
                  <div>
                    <span className="font-medium">Total Messages:</span> {modelPerformance.conversation_stats?.total_messages || 0}
                  </div>
                  <div>
                    <span className="font-medium">Healthcare Keywords:</span> {modelPerformance.healthcare_domain_coverage?.keywords_covered || 0}
                  </div>
                  <div>
                    <span className="font-medium">Intent Categories:</span> {modelPerformance.healthcare_domain_coverage?.intent_categories || 0}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}


