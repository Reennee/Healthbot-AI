const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatMessage {
  message: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  confidence: number;
  domain_relevance: boolean;
}

export interface HealthQuery {
  query: string;
  context?: string;
}

export interface SymptomAnalysis {
  analysis: string;
  recommendation: string;
  urgency: "low" | "medium" | "high";
  domain_relevance: boolean;
}

class HealthBotAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async chat(message: string, conversationId?: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Chat API error:", error);
      throw error;
    }
  }

  async analyzeSymptoms(
    query: string,
    context?: string
  ): Promise<SymptomAnalysis> {
    try {
      const response = await fetch(`${this.baseUrl}/analyze-symptoms`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          context,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Symptom analysis API error:", error);
      throw error;
    }
  }

  async getConversation(conversationId: string) {
    try {
      const response = await fetch(
        `${this.baseUrl}/conversation/${conversationId}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Get conversation API error:", error);
      throw error;
    }
  }

  async deleteConversation(conversationId: string) {
    try {
      const response = await fetch(
        `${this.baseUrl}/conversation/${conversationId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Delete conversation API error:", error);
      throw error;
    }
  }

  async getModelInfo() {
    try {
      const response = await fetch(`${this.baseUrl}/model/info`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Model info API error:", error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Health check API error:", error);
      throw error;
    }
  }
}

export const healthBotAPI = new HealthBotAPI();
